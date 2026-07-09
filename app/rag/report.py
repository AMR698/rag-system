"""
Final report generator.

All quantitative numbers (score, accuracy, average_time, highest_level,
question_analysis) are computed deterministically from the in-memory
session records — never by the LLM, to guarantee correctness.

The LLM (via app/prompts/report_prompt.py) is used ONLY to produce the
qualitative narrative sections: strengths, weaknesses, common_mistakes,
and recommendations, grounded in the deterministic summary.
"""

from __future__ import annotations

import logging
from typing import List

from app.prompts.report_prompt import build_report_prompt
from app.rag.llm_client import LLMClientError, call_llm, safe_json_parse
from app.schemas import AnsweredQuestionRecord, Difficulty, FinalReport, QuestionAnalysis

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an educational performance analyst. You always answer in JSON, "
    "grounded strictly in the data given to you."
)

_DIFFICULTY_ORDER = {Difficulty.EASY: 0, Difficulty.MEDIUM: 1, Difficulty.HARD: 2}

# Safe fallback narrative used only if the LLM call fails.
_FALLBACK_NARRATIVE = {
    "strengths": ["Completed the session and engaged with multiple questions."],
    "weaknesses": ["Not enough data to identify a specific weakness."],
    "common_mistakes": ["No clear pattern detected in this session."],
    "recommendations": ["Keep practicing regularly to build confidence."],
}


class ReportGenerator:
    """Builds the final performance report for a finished session."""

    @staticmethod
    def _highest_level(records: List[AnsweredQuestionRecord]) -> Difficulty:
        if not records:
            return Difficulty.EASY
        return max((r.difficulty for r in records), key=lambda d: _DIFFICULTY_ORDER[d])

    def build_summary(self, records: List[AnsweredQuestionRecord]) -> dict:
        """Compute all deterministic statistics from the session records."""
        total = len(records)
        correct = sum(1 for r in records if r.is_correct)
        wrong = total - correct
        accuracy = round((correct / total) * 100, 2) if total else 0.0
        average_time = round(sum(r.time_seconds for r in records) / total, 2) if total else 0.0
        student_score = round((correct * 10 / total) * 10, 0) if total else 0  # 0-100 scale
        highest_level = self._highest_level(records)

        question_analysis = [
            QuestionAnalysis(
                question=r.question,
                difficulty=r.difficulty,
                time=r.time_seconds,
                correct=r.is_correct,
            )
            for r in records
        ]

        return {
            "student_score": int(student_score),
            "correct_answers": correct,
            "wrong_answers": wrong,
            "accuracy": accuracy,
            "average_time": average_time,
            "highest_level": highest_level.value,
            "question_analysis": [qa.model_dump() for qa in question_analysis],
        }

    def generate(self, records: List[AnsweredQuestionRecord]) -> FinalReport:
        """
        Build the complete final report: deterministic stats + LLM-
        generated narrative sections.
        """
        summary = self.build_summary(records)

        narrative = dict(_FALLBACK_NARRATIVE)
        try:
            prompt = build_report_prompt(summary)
            raw_response = call_llm(prompt=prompt, system_prompt=SYSTEM_PROMPT, temperature=0.5)
            parsed = safe_json_parse(raw_response)
            narrative = {
                "strengths": parsed.get("strengths", _FALLBACK_NARRATIVE["strengths"]),
                "weaknesses": parsed.get("weaknesses", _FALLBACK_NARRATIVE["weaknesses"]),
                "common_mistakes": parsed.get("common_mistakes", _FALLBACK_NARRATIVE["common_mistakes"]),
                "recommendations": parsed.get("recommendations", _FALLBACK_NARRATIVE["recommendations"]),
            }
        except LLMClientError as exc:
            logger.warning("Falling back to default report narrative due to LLM error: %s", exc)

        return FinalReport(
            student_score=summary["student_score"],
            correct_answers=summary["correct_answers"],
            wrong_answers=summary["wrong_answers"],
            accuracy=summary["accuracy"],
            average_time=summary["average_time"],
            highest_level=Difficulty(summary["highest_level"]),
            strengths=narrative["strengths"],
            weaknesses=narrative["weaknesses"],
            common_mistakes=narrative["common_mistakes"],
            recommendations=narrative["recommendations"],
            question_analysis=[QuestionAnalysis(**qa) for qa in summary["question_analysis"]],
        )
