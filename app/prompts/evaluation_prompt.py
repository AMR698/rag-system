"""
Prompt template used to instruct the LLM (Qwen) to evaluate a player's
answer and produce structured feedback.

Note: correctness itself (is_correct) is determined deterministically
in code (string comparison) by the evaluator, NOT by the LLM — this
avoids hallucinated grading. The LLM is only used to generate the
short natural-language feedback message and a 0-10 quality score.
"""


def build_evaluation_prompt(
    question: str,
    correct_answer: str,
    player_answer: str,
    is_correct: bool,
    difficulty: str,
    time_taken: float,
) -> str:
    """
    Build the prompt sent to the LLM to produce feedback text and a
    score for a single answered question.

    Args:
        question: The original question text.
        correct_answer: The verified correct answer.
        player_answer: What the player actually answered.
        is_correct: Deterministically computed correctness flag.
        difficulty: Difficulty level of the question.
        time_taken: Time (seconds) the player took to answer.

    Returns:
        A single formatted prompt string.
    """
    return f"""You are a friendly, encouraging tutor for young learners.

A player answered a question in an educational game. Here are the facts:

Question: "{question}"
Difficulty: "{difficulty}"
Correct answer: "{correct_answer}"
Player answer: "{player_answer}"
Time taken: {time_taken} seconds
Correctness (already verified, do not change this): {is_correct}

Your task:
1. Write ONE short, warm, age-appropriate feedback sentence (max 20 words) for the player.
   - If correct: be encouraging and specific.
   - If incorrect: be kind, mention the correct answer gently, and encourage retrying.
2. Give a "score" integer from 0 to 10 reflecting answer quality and speed
   (10 = correct and fast, 0 = incorrect).

STRICT RULES:
- Do NOT change or contradict the "Correctness" value given above.
- Output ONLY valid JSON, no markdown, no extra text.

Return EXACTLY this JSON schema:
{{
  "score": 0,
  "feedback": "string"
}}
"""
