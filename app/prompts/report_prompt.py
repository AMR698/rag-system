"""
Prompt template used to instruct the LLM (Qwen) to generate the
qualitative parts of the final performance report:
strengths, weaknesses, common_mistakes, recommendations.

All quantitative fields (score, accuracy, average_time, etc.) are
computed deterministically in code (see app/rag/report.py) and are
passed into this prompt only as context — the LLM never invents
numbers, it only reasons about them in natural language.
"""

import json


def build_report_prompt(session_summary: dict) -> str:
    """
    Build the prompt sent to the LLM to generate the narrative sections
    of the final report.

    Args:
        session_summary: A dict containing the deterministic stats and
            the full list of answered questions for this session.

    Returns:
        A single formatted prompt string.
    """
    summary_json = json.dumps(session_summary, ensure_ascii=False, indent=2)

    return f"""You are an educational performance analyst reviewing a student's game session.

Below is the deterministic session summary (already computed, do not recompute or contradict any numbers):

{summary_json}

Your task is to analyze this data and produce ONLY the following four lists:
1. "strengths": 2-4 short bullet points about what the student did well.
2. "weaknesses": 2-4 short bullet points about where the student struggled.
3. "common_mistakes": 1-3 short bullet points describing patterns in wrong answers (e.g. topic, difficulty level, timing).
4. "recommendations": 2-4 short, actionable, encouraging bullet points for what to practice next.

STRICT RULES:
- Base every point ONLY on the data provided above. Do not invent facts about the student.
- Keep each bullet point short (max 15 words), simple, and age-appropriate.
- Output ONLY valid JSON. No markdown, no code fences, no extra text.

Return EXACTLY this JSON schema:
{{
  "strengths": ["string", "..."],
  "weaknesses": ["string", "..."],
  "common_mistakes": ["string", "..."],
  "recommendations": ["string", "..."]
}}
"""
