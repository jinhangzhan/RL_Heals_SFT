import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

SYSTEM_PROMPT1 = """
You are required to respond in **exactly** the following format, with no extra text before or after:

<reasoning>
[Your entire step-by-step reasoning goes here]
</reasoning>
<answer>
[Final numeric or concise answer goes here]
</answer>

---
**Example**  
**Question**: What is 5 + 3?

Correct response:

<reasoning>
First, note that 5 + 3 = 8.
</reasoning>
<answer>
8
</answer>

---
**Instructions**:
1. Do not add any text **before** <reasoning>.
2. Do not add any text **after** </answer>.
3. Do not repeat or redefine the tags. They must appear **once** each.
4. Place all step-by-step thoughts inside <reasoning>...</reasoning>.
5. Place your final, concise answer inside <answer>...</answer>.
6. Do not include “The final answer is …” or any extra words outside the tags.  

Now, please answer the question below in this exact format.
"""


def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


def get_gsm8k_questions(split="test") -> Dataset:
    # data = load_dataset('openai/gsm8k', 'main')[split]
    data = load_dataset('/mnt/workspace/jinhangzhan/rl/gsm8k', 'main')[split]

    def map_example(x):
        question = x["question"]
        gold = extract_hash_answer(x["answer"])
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}
            ],
            "answer": gold
        }

    data = data.map(map_example)
    return data


def build_chat_text(messages) -> str:
    text = ""
    for m in messages:
        role = m["role"]
        content = m["content"].strip()
        if role == "system":
            text += f"{content}\n"
        elif role == "user":
            text += f"{content}\n"
    return text


def evaluate_accuracy(model: AutoModelForCausalLM, tokenizer, dataset: Dataset) -> float:
    device = model.device
    model.eval()

    correct = 0
    total = 0

    for idx, example in enumerate(dataset):
        prompt_messages = example["prompt"]
        gold_answer = example["answer"]  # e.g. "7"
        if gold_answer is None:
            continue

        prompt_text = build_chat_text(prompt_messages)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.0,  # greedy
            )

        pred_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print('prompt=======', prompt_text)
        print('answer=======', pred_text)
        pred_answer = extract_xml_answer(pred_text)

        if pred_answer == gold_answer:
            correct += 1
        total += 1

        if (idx + 1) % 1 == 0:
            print(f"[{idx + 1}] Gold: {gold_answer} | Pred: {pred_answer[:50]}")

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def main():
    eval_split = "test"
    dataset = get_gsm8k_questions(split=eval_split)

    model_checkpoint = "/mnt/workspace/jinhangzhan/data/Qwen2.5-1.5B-Instruct"
    model_checkpoint = "/mnt/workspace/jinhangzhan/rl/outputs/Qwen-1.5B-GRPO/checkpoint-7473"

    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    accuracy = evaluate_accuracy(model, tokenizer, dataset)
    print(f"\nEvaluation on GSM8k {eval_split} set done! Accuracy = {accuracy:.4f}")


if __name__ == "__main__":
    main()