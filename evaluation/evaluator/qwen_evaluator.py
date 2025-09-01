import os
from PIL import Image
import torch
import torch.optim as optim
from accelerate.state import AcceleratorState
from transformers import AutoTokenizer
from .base_evaluator import BaseEvaluator
from utils_mllm import evaluate_model_config

class QwenEvaluator(BaseEvaluator):
    def __init__(self, action_space, daytime, env_config, model, model_path,
                 prompt_config, generation_config, output_dir, seed=42, **kwargs):
        super(QwenEvaluator, self).__init__(
            action_space, daytime, env_config, prompt_config,
            generation_config, output_dir, seed, **kwargs
        )

        self.tokenizer, model = evaluate_model_config(model, model_path)
        dummy_optim = optim.Adam(model.parameters(), lr=1e-4)

        ds_plugin = AcceleratorState().deepspeed_plugin
        if ds_plugin is not None:
            ds_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = 1

        self.model, _ = self.accelerator.prepare(model, dummy_optim)

    # --------- prompt / payload -------------------------------------------------
    def formulate_payload(self, question):
        # self.payload = [{
        #     "role": "user",
        #     "content": [{"type": "text", "text": question}]
        # }]
        self.payload = [{"role": "user", "content": question}]

    def formulate_prompt(self, vision_res_dict, language_res_dict,
                         prompt=None, info=None):
        if "gym_cards" in self.id:
            if info["Verify Info"] is not None:
                self.payload.append({
                    "role": "user",
                    "content": f"You failed this trial because {info['Verify Info']}"
                })
            else:
                question = prompt.format(**vision_res_dict,
                                          **language_res_dict,
                                          **self.oracle_arguments)
                self.formulate_payload(question)

        elif "gym_virl" in self.id:
            if info["Verify Info"] is None or "Correct action" in info["Verify Info"] \
               or "step_limit_reached" in info["Verify Info"]:
                question = prompt.format(**vision_res_dict,
                                          **language_res_dict,
                                          **self.oracle_arguments)
                self.formulate_payload(question)
            else:
                self.payload.append({
                    "role": "user",
                    "content": f"You failed this trial because {info['Verify Info']}"
                })

    # --------- generation -------------------------------------------------------
    def generate_one_response(self, vision_res_dict, language_res_dict,
                              obs=None, info=None):
        if obs is not None:
            raise ValueError("QwenEvaluator does not support vision input.")

        self.formulate_vision_arguments(vision_res_dict, info)
        prompts, patterns = self.prompt_language, self.pattern_language

        for prompt, pattern in zip(prompts, patterns):
            self.formulate_prompt(vision_res_dict, language_res_dict,
                                  prompt, info=info)

            # chat_text = self.tokenizer.apply_chat_template(
            #     self.payload, add_generation_prompt=True
            # )
            chat_text = self.tokenizer.apply_chat_template(
                self.payload,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = self.tokenizer(chat_text,
                                    return_tensors="pt",
                                    add_special_tokens=False).to(
                                        self.model.device)

            output_ids = self.model.generate(**inputs, **self.generation_config)
            response = self.tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            language_res_dict[pattern] = response
            self.append_intermidiate_res(response)

    # --------- helpers ----------------------------------------------------------
    def append_intermidiate_res(self, res):
        self.payload.append({"role": "assistant", "content": res})

    def extract_final_action(self, language_res_dict, key_pattern=None):
        try:
            return language_res_dict[key_pattern]
        except KeyError:
            if "gym_cards" in self.id:
                return language_res_dict["formula"]
            return language_res_dict["action"]