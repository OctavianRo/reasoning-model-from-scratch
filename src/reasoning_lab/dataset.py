"""Synthetic reasoning tasks with verifiable answers."""

from __future__ import annotations

from dataclasses import dataclass
import random


NAMES = ["Maya", "Noah", "Ava", "Liam", "Sofia", "Ethan"]
OBJECTS = ["marbles", "stickers", "cards", "coins", "pencils", "shells"]


@dataclass(frozen=True)
class Problem:
    """A small word problem with a ground-truth reasoning trace."""

    question: str
    start: int
    add: int
    subtract: int
    answer: int

    @property
    def gold_trace(self) -> str:
        after_add = self.start + self.add
        return (
            f"Start with {self.start}. "
            f"Add {self.add} to get {after_add}. "
            f"Subtract {self.subtract} to get {self.answer}."
        )


def make_problem(rng: random.Random) -> Problem:
    """Create one arithmetic word problem."""

    start = rng.randint(3, 18)
    add = rng.randint(1, 12)
    max_subtract = max(1, start + add - 1)
    subtract = rng.randint(1, max_subtract)
    answer = start + add - subtract
    name = rng.choice(NAMES)
    obj = rng.choice(OBJECTS)
    question = (
        f"{name} has {start} {obj}, buys {add} more, "
        f"then gives away {subtract}. How many {obj} remain?"
    )
    return Problem(
        question=question,
        start=start,
        add=add,
        subtract=subtract,
        answer=answer,
    )


def make_dataset(size: int, seed: int = 7) -> list[Problem]:
    """Create a deterministic dataset for experiments and interviews."""

    rng = random.Random(seed)
    return [make_problem(rng) for _ in range(size)]

