"""
Prompt template used to instruct the LLM (Qwen) to generate a single
multiple-choice question strictly from the retrieved lesson context.

Kept in its own file (per project spec) so prompt engineering can be
iterated on without touching any application logic.
"""

INSUFFICIENT_CONTEXT_MARKER = "Insufficient lesson information."


def build_question_prompt(lesson_context: str, difficulty: str, previous_questions: list[str] | None = None) -> str:
    """
    Build the full prompt sent to the LLM to generate one MCQ question.

    Args:
        lesson_context: The merged text chunks retrieved from ChromaDB
            for the selected lesson_id ONLY.
        difficulty: One of "easy", "medium", "hard". The LLM is forced
            to generate a question that matches this exact level.
        previous_questions: Optional list of questions already asked in
            this session, so the LLM avoids repeating them.

    Returns:
        A single formatted prompt string.
    """
    previous_questions = previous_questions or []
    previous_block = (
        "\n".join(f"- {q}" for q in previous_questions[-10:])
        if previous_questions
        else "None yet."
    )

    return f"""You are an expert educational content generator for a KG/school learning game.

STRICT RULES (do not break any of them):
1. You must generate the question ONLY using the information inside the "LESSON CONTEXT" section below.
2. Never use your own external knowledge, facts, or assumptions that are not present in the lesson context.
3. If the lesson context below is empty, too short, or not enough to build a fair question, respond with EXACTLY this text and nothing else:
{INSUFFICIENT_CONTEXT_MARKER}
4. The question difficulty MUST match exactly: "{difficulty}".
   - easy: simple recall / direct fact from the lesson.
   - medium: requires connecting two ideas from the lesson.
   - hard: requires applying or reasoning about the lesson's concept.
5. Do NOT repeat any of the previously asked questions listed below.
6. Output ONLY valid JSON. No markdown, no code fences, no explanation, no extra text before or after the JSON.

PREVIOUSLY ASKED QUESTIONS (avoid repeating these):
{previous_block}

LESSON CONTEXT:
\"\"\"
{lesson_context}
\"\"\"

Return the question using EXACTLY this JSON schema:
{{
  "type": "mcq",
  "question": "string",
  "choices": ["string", "string", "string", "string"],
  "correct_answer": "string (must be one of the choices, verbatim)",
  "difficulty": "{difficulty}"
}}
"""
