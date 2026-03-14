import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces
from collections import namedtuple
from utils_rl import REWARD_FN


class RankEnv(gym.Env):
    """
    Gym environment for matrix rank problems.

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

        # Map difficulty to generator params (per user spec)
        if self.difficulty == 1:
            rows, cols = 4, 5
            min_val, max_val = -4, 4
            rank = random.randint(0, 4)
        else:  # difficulty 2 (OOD)
            rows, cols = 5, 4
            min_val, max_val = -5, 5
            rank = random.randint(0, 4)

        self.problem = _generate_matrix_rank_problem(
            rows=rows, cols=cols, min_val=min_val, max_val=max_val, specific_rank=rank
        )

        # Build prompt-like text; keep both Question and Matrix fields for trainers
        question_text = (
            "Find the rank of the matrix.\n" + self.problem.question.split('\n', 1)[-1]
        )
        import re
        matrix_match = re.search(r'\[((\[.*?\])(,\n \[.*?\])*)\]', question_text)
        matrix_text = matrix_match.group(0) if matrix_match else None
        # Keep answer as int; align with det env modulo-1000 comparison
        answer = int(round(self.problem.answer)) % 1000

        obs = np.zeros((self.resolution, self.resolution, 3), dtype=np.uint8)
        self.info = {
            "Question": question_text,
            "Matrix": matrix_text,
            "Solution": [str(answer)],
            "Verify Info": None,
            "Remaining Step": self.remaining_step,
        }
        return obs, self.info

    def step(self, action_text: str):
        # Verification-style multi-step task
        self.remaining_step -= 1

        pred_answer = self._extract_answer(action_text)
        if pred_answer is None:
            reward = REWARD_FN["NO_SOLUTION"]
            verify_msg = "failed_to_parse_answer"
        else:
            gt = int(self.info["Solution"][0])
            if int(pred_answer) % 1000 == gt % 1000:
                reward = REWARD_FN["CORRECT_SOLUTION"]
                verify_msg = "Correct solution"
            else:
                reward = REWARD_FN["NO_SOLUTION"]
                verify_msg = "incorrect_answer"

        if reward == REWARD_FN["CORRECT_SOLUTION"]:
            done, truncated = True, False
        else:
            if self.remaining_step < 0:
                done, truncated = False, True
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
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "answer" in obj:
                return int(str(obj["answer"]).strip())
        except Exception:
            pass
        m = re.search(r'"answer"\s*:\s*(-?\d+)', text)
        if m:
            return int(m.group(1))
        m2 = re.search(r'(-?\d+)', text)
        if m2:
            return int(m2.group(1))
        return None


# ---------------- Self-contained generators (no external imports) ---------------- #

Problem = namedtuple('Problem', ('question', 'answer'))


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


def _matrix_rank_int(M):
    # numpy rank with default tol is fine; keep deterministic enough for ints
    return int(np.linalg.matrix_rank(M))


def _generate_matrix_with_rank(rows, cols, rank, min_val, max_val):
    if rank == 0:
        return np.zeros((rows, cols), dtype=int)
    # Ensure rank <= min(rows, cols)
    rank = min(rank, rows, cols)
    # Try to construct A (rows x rank), B (rank x cols) both full-rank
    for _ in range(2000):
        A = np.random.randint(min_val, max_val + 1, size=(rows, rank))
        B = np.random.randint(min_val, max_val + 1, size=(rank, cols))
        if _matrix_rank_int(A) == rank and _matrix_rank_int(B) == rank:
            M = A @ B
            if _matrix_rank_int(M) == rank:
                return M
    # Fallback: brute force until match
    for _ in range(5000):
        M = np.random.randint(min_val, max_val + 1, size=(rows, cols))
        if _matrix_rank_int(M) == rank:
            return M
    # Last resort: return something with at most target rank
    return A @ B


def _generate_matrix_rank_problem(rows=4, cols=5, min_val=-4, max_val=4, specific_rank=2):
    M = _generate_matrix_with_rank(rows, cols, specific_rank, min_val, max_val)
    matrix_str = _format_matrix(M)
    question = "Matrix:\n" + matrix_str
    answer = _matrix_rank_int(M)
    return Problem(question=question, answer=answer)
