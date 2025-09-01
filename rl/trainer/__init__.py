# from rl.trainer.cambrian_rl_trainer import *
# from rl.trainer.llama_rl_trainer import *
from rl.trainer.llama_trainer import *
from rl.trainer.qwen_trainer import *
trainer_init = {
    # 'Cambrian_RL_Trainer': Cambrian_RL_Trainer,
    # 'Llama_RL_Trainer': Llama_RL_Trainer_oneline,
    'LlamaTrainer': LlamaTrainer,
    'QwenTrainer': QwenTrainer
}