# from evaluation.evaluator.api_evaluator import *
# from evaluation.evaluator.oai_evaluator import *
# from evaluation.evaluator.oai_evaluator_oneline import *
# # from evaluator.cambrian_evaluator import *
# from evaluation.evaluator.cambrian_evaluator_oneline import *
# from evaluation.evaluator.llama_vl_evaluator_oneline import *
from evaluation.evaluator.llama_evaluator import LlamaEvaluator
from evaluation.evaluator.qwen_evaluator import QwenEvaluator
evaluator_init = {
    # "ApiEvaluator": ApiEvaluator,
    # "OpenAIEvaluator": OpenAIEvaluator,
    # "OpenAIEvaluator_oneline": OpenAIEvaluator_oneline,
    # # "CambrianEvaluator": CambrianEvaluator,
    # "CambrianEvaluator_oneline": CambrianEvaluator_oneline,
    # "LlamaEvaluator_oneline": LlamaEvaluator_oneline,
    "LlamaEvaluator": LlamaEvaluator,
    "QwenEvaluator": QwenEvaluator
}