"""
Answer evaluator.

Correctness is computed deterministically in code (case-insensitive,
whitespace-trimmed string comparison) — we never trust an LLM to judge
right/wrong, since that could hallucinate. The LLM is used only to
generate a short, encouraging feedback message and a 0-10 quality
score that also factors in response time.
"""

from __future__ import annotations

import logging

from app.prompts.evaluation_prompt import build_evaluation_prompt
from app.rag.llm_client import LLMClientError, call_llm, safe_json_parse
from app.schemas import Difficulty, EvaluationResult

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a kind, encouraging tutor for young learners. You always answer in JSON."
)

# Fallback feedback used if the LLM call fails, so the game never breaks.
_FALLBACK_FEEDBACK_CORRECT = "Great job, that's correct!"
_FALLBACK_FEEDBACK_WRONG = "Not quite — keep trying, you're learning!"


class AnswerEvaluator:
    """Evaluates a single answer and returns a structured EvaluationResult."""

    @staticmethod
    def _is_correct(correct_answer: str, player_answer: str) -> bool:
        return correct_answer.strip().casefold() == player_answer.strip().casefold()

    def evaluate(
        self,
        question: str,
        correct_answer: str,
        player_answer: str,
        difficulty: Difficulty,
        time_taken: float,
    ) -> EvaluationResult:
        is_correct = self._is_correct(correct_answer, player_answer)

        prompt = build_evaluation_prompt(
            question=question,
            correct_answer=correct_answer,
            player_answer=player_answer,
            is_correct=is_correct,
            difficulty=difficulty.value,
            time_taken=time_taken,
        )

        score = 10 if is_correct else 0
        feedback = _FALLBACK_FEEDBACK_CORRECT if is_correct else _FALLBACK_FEEDBACK_WRONG

        try:
            raw_response = call_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.6)
            parsed = safe_json_parse(raw_response)
            score = int(parsed.get("score", score))
            feedback = str(parsed.get("feedback", feedback))
        except (LLMClientError, ValueError, TypeError) as exc:
            # Never let a feedback-generation failure break the game loop.
            logger.warning("Falling back to default feedback due to LLM error: %s", exc)

        return EvaluationResult(
            is_correct=is_correct,
            score=max(0, min(10, score)),
            difficulty=difficulty,
            time=time_taken,
            feedback=feedback,
        )
