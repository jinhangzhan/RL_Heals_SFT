import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple
from utils_rl import REWARD_FN


class DeterminantEnv(gym.Env):
    """
    Gym environment for determinant problems.

    Observation: an RGB canvas (placeholder) for compatibility.
    Action: a JSON/text string containing an "answer" integer.
    Reward: +REWARD_FN["CORRECT_SOLUTION"] if correct; otherwise REWARD_FN["NO_SOLUTION"].

    Args (via gym.make kwargs):
      - difficulty: 1 (ID) or 2 (OOD). Default 1.
      - verify_iter: number of verify steps (default: 1; single-step task).
      - seed: RNG seed.
      - ood: bool, if True prefer difficulty=2.
      - language_only: bool, unused but kept for compatibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, difficulty: int = 1, verify_iter: int = 1, ood: bool = False,
                 resolution: int = 256, language_only: bool = True, **kwargs):
        super().__init__()
        self.difficulty = 2 if ood else difficulty
        self.verify_iter = verify_iter
        self.resolution = resolution
        self.language_only = language_only
        self._seeded_once = False

        # Placeholder observation to fit existing pipelines
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(resolution, resolution, 3), dtype=np.uint8
        )
        # Single text action; upstream uses text generation, not this space
        self.action_space = spaces.Discrete(1)

        # Internals
        self._remaining = 0
        self.problem = None
        self.matrix = None
        self.info = {}

    def seed(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        # Seed only on the first reset to avoid regenerating the exact same
        # sample across episodes when caller passes the same seed each time.
        if (seed is not None) and (not self._seeded_once):
            self.seed(seed)
            self._seeded_once = True
        self.remaining_step = self.verify_iter

        # Map difficulty to generator params
        if self.difficulty == 1:
            rows = 2
            min_val, max_val = -4, 4
            special = random.choice([None, 'symmetric'])
        else:  # difficulty 2 (OOD)
            rows = 3
            min_val, max_val = -5, 5
            special = random.choice([None, 'symmetric'])

        # Generate one determinant problem (self-contained; no external imports)
        self.problem = _generate_matrix_determinant_problem(
            rows=rows, special_type=special, min_val=min_val, max_val=max_val
        )

        # Extract a clean matrix render for prompts
        # Regenerate matrix text by parsing from question if present
        # Fallback: just show rows x rows matrix from reformatting
        # We cannot recover the exact matrix from question string reliably here,
        # so we expose question text directly and add a separate formatted string.
        import re
        self.question_text = self.problem.question
        matrix_match = re.search(r'\[((\[.*?\])(,\n \[.*?\])*)\]', self.question_text)
        matrix_text = matrix_match.group(0) if matrix_match else None
        self.answer = int(self.problem.answer)

        obs = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self.info = {
            "Question": self.question_text,
            "Matrix": matrix_text,
            "Solution": [str(self.answer)],  # compatible with downstream logging
            "Verify Info": None,
            "Remaining Step": self.remaining_step,
        }
        return obs, self.info

    def step(self, action_text: str):
        # Verification-style multi-step task
        self.remaining_step -= 1

        # Extract integer answer from model output
        pred_answer = self._extract_answer(action_text)
        if pred_answer is None:
            reward = REWARD_FN["NO_SOLUTION"]
            verify_msg = "failed_to_parse_answer"
        else:
            # Compare raw determinant (no modulo)
            if int(pred_answer) == self.answer:
                reward = REWARD_FN["CORRECT_SOLUTION"]
                verify_msg = "Correct solution"
            else:
                reward = REWARD_FN["NO_SOLUTION"]
                verify_msg = "incorrect_answer"

        # Control episode termination by verify_iter
        if reward == REWARD_FN["CORRECT_SOLUTION"]:
            done, truncated = True, False
        else:
            if self.remaining_step < 0:
                done, truncated = False, True  # step limit reached
            else:
                done, truncated = False, False
        obs = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        info = {
            **self.info,
            "Pred Answer": pred_answer,
            "Verify Info": verify_msg,
            "Remaining Step": self.remaining_step,
        }
        return obs, reward, done, truncated, info

    @staticmethod
    def _extract_answer(text: str):
        """Try to parse integer after key "answer" in JSON or free text."""
        import json, re
        # JSON path
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "answer" in obj:
                return int(str(obj["answer"]).strip())
        except Exception:
            pass
        # Fallback regex
        m = re.search(r'"answer"\s*:\s*(-?\d+)', text)
        if m:
            return int(m.group(1))
        # Final fallback: bare integer in text
        m2 = re.search(r'(-?\d+)', text)
        if m2:
            return int(m2.group(1))
        return None


# ---------------- Self-contained generators (no external imports) ---------------- #

Problem = namedtuple('Problem', ('question', 'answer'))

def _generate_matrix(rows, cols, min_val=-5, max_val=5, integer_only=True, special_type=None):
    if integer_only:
        M = np.random.randint(min_val, max_val + 1, size=(rows, cols))
    else:
        M = np.random.uniform(min_val, max_val, size=(rows, cols))
    if special_type == 'symmetric':
        M = (M + M.T) / 2
        if integer_only:
            M = np.round(M).astype(int)
    return M

def _format_matrix(matrix, decimals=2):
    r, c = matrix.shape
    out = "["
    for i in range(r):
        out += "["
        for j in range(c):
            val = matrix[i, j]
            if np.isclose(val, round(val)):
                out += f"{int(round(val))}"
            else:
                out += f"{val:.{decimals}f}"
            if j < c - 1:
                out += ", "
        out += "]"
        if i < r - 1:
            out += ",\n "
    out += "]"
    return out

def _generate_matrix_determinant_problem(rows=None, special_type=None, min_val=-5, max_val=5):
    if rows is None:
        rows = random.choice([2, 3, 4])

    # Try until a reasonably integer determinant is found and non-zero
    for _ in range(1000):
        M = _generate_matrix(rows, rows, min_val, max_val, integer_only=True, special_type=special_type)
        det_real = float(np.linalg.det(M))
        det_rounded = int(round(det_real))
        if abs(det_real - det_rounded) < 1e-8 and det_rounded != 0:
            matrix_str = _format_matrix(M)
            question = (
                "Calculate the determinant of the matrix:\n" + matrix_str
            )
            answer = det_rounded
            return Problem(question=question, answer=answer)

    # Fallback: just return last attempt even if not ideal (should rarely happen)
    matrix_str = _format_matrix(M)
    question = "Calculate the determinant of the matrix:\n" + matrix_str
    return Problem(question=question, answer=int(round(det_real)))
