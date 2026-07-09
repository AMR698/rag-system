"""
Question generator.

Takes the retrieved lesson context + the current adaptive difficulty
level, builds the prompt (from app/prompts/question_prompt.py), calls
the LLM, and returns a validated QuestionModel.

The LLM is instructed to reply with the exact string
"Insufficient lesson information." whenever the retrieved context is
not enough to build a fair question — this module detects that marker
and raises a dedicated exception so the API layer can return a clean
error instead of a broken question.
"""

from __future__ import annotations

import logging

from app.prompts.question_prompt import INSUFFICIENT_CONTEXT_MARKER, build_question_prompt
from app.rag.llm_client import LLMClientError, call_llm, safe_json_parse
from app.schemas import Difficulty, QuestionModel

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a strict educational question generator. You only use the "
    "context given to you. You never invent facts. You always answer in JSON."
)


class InsufficientContextError(Exception):
    """Raised when the LLM reports the lesson context is insufficient."""


class GenerationError(Exception):
    """Raised when the LLM response cannot be turned into a valid question."""


class QuestionGenerator:
    """Generates a single adaptive-difficulty MCQ question from a lesson context."""

    def generate(
        self,
        lesson_context: str,
        difficulty: Difficulty,
        previous_questions: list[str] | None = None,
    ) -> QuestionModel:
        """
        Args:
            lesson_context: Merged text chunks retrieved for the lesson.
            difficulty: Current adaptive difficulty level for the session.
            previous_questions: Questions already asked, to avoid repeats.

        Raises:
            InsufficientContextError: if the LLM determines the lesson
                context is not enough to build a question.
            GenerationError: if the LLM output is malformed / invalid.
        """
        if not lesson_context or not lesson_context.strip():
            # Fail fast without even calling the LLM if there is literally no context.
            raise InsufficientContextError(INSUFFICIENT_CONTEXT_MARKER)

        prompt = build_question_prompt(
            lesson_context=lesson_context,
            difficulty=difficulty.value,
            previous_questions=previous_questions,
        )

        try:
            raw_response = call_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)
        except LLMClientError as exc:
            raise GenerationError(str(exc)) from exc

        if INSUFFICIENT_CONTEXT_MARKER in raw_response:
            raise InsufficientContextError(INSUFFICIENT_CONTEXT_MARKER)

        try:
            parsed = safe_json_parse(raw_response)
            question = QuestionModel(
                type=parsed.get("type", "mcq"),
                question=parsed["question"],
                choices=parsed["choices"],
                correct_answer=parsed["correct_answer"],
                difficulty=Difficulty(parsed.get("difficulty", difficulty.value)),
            )
        except (LLMClientError, KeyError, ValueError) as exc:
            logger.error("Malformed question payload from LLM: %s", raw_response)
            raise GenerationError(f"LLM returned an invalid question payload: {exc}") from exc

        # Defensive validation: correct_answer must actually be one of the choices.
        if question.correct_answer not in question.choices:
            logger.error("correct_answer not found in choices: %s", parsed)
            raise GenerationError("LLM's correct_answer is not among the provided choices.")

        return question
