from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

conv_mode = "llama_3" 

def evaluate_model_config(model, model_path, device="cuda"):
    if "cambrian" in model:
        raise NotImplementedError
    elif "llama" in model:
        processor = AutoProcessor.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path)
        model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
            )
        return processor, model
    elif "qwen" in model:
        # processor = AutoProcessor.from_pretrained(model_path)

        processor = AutoTokenizer.from_pretrained(model_path, padding_side="right")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
        )
        return processor, model
                