from .prompts import *

PROMPT_FN = {k: globals()[k] for k in globals() if k.startswith("Q_") or k.startswith("Response")}