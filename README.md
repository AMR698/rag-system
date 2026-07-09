# Educational RAG Game Backend

A production-ready **Retrieval-Augmented Generation (RAG) module** for an
educational game. This repository contains **only the RAG backend** —
no frontend, no authentication, no gameplay UI.

It connects to an **already-embedded** ChromaDB collection, retrieves
content scoped to a single `lesson_id`, generates adaptive-difficulty
multiple-choice questions with the Qwen LLM, evaluates player answers,
tracks the session in memory, and produces a final JSON performance
report.

---

## 1. Architecture Overview

```
Frontend (not part of this repo)
        │  lesson_id
        ▼
 POST /start-session ──────────────► retrieves ALL chunks for lesson_id
        │                             from ChromaDB (metadata filter)
        ▼
 POST /generate-question ──────────► builds prompt from lesson context
        │                             + current difficulty → Qwen LLM
        ▼
 POST /submit-answer ───────────────► deterministic correctness check
        │                             + Qwen-generated feedback
        │                             + adaptive difficulty update
        ▼
 POST /finish-session ──────────────► deterministic stats + Qwen-generated
                                        strengths/weaknesses/recommendations
                                        → pure JSON report
```

### Folder Structure

```
rag_system/
├── app/
│   ├── api/
│   │   └── routes.py           # FastAPI endpoints
│   ├── rag/
│   │   ├── retriever.py        # Loads existing Chroma collection, lesson-scoped retrieval
│   │   ├── generator.py        # Adaptive MCQ question generation
│   │   ├── evaluator.py        # Deterministic correctness + LLM feedback
│   │   ├── report.py           # Deterministic stats + LLM narrative report
│   │   └── llm_client.py       # Shared Qwen (OpenAI-compatible) client wrapper
│   ├── services/
│   │   └── game_service.py     # Session lifecycle + adaptive difficulty state machine
│   ├── config/
│   │   └── settings.py         # Loads all config from secrets/.env
│   ├── prompts/
│   │   ├── question_prompt.py
│   │   ├── evaluation_prompt.py
│   │   └── report_prompt.py
│   ├── schemas.py               # Shared Pydantic request/response models
│   └── main.py                  # FastAPI app entrypoint
├── vector_db/
│   └── chroma_store/            # Existing persistent Chroma DB (already embedded elsewhere)
├── secrets/
│   └── .env                     # Environment variables (never hardcode secrets)
├── requirements.txt
└── README.md
```

---

## 2. Prerequisites

- Python 3.12
- An **already-populated** ChromaDB persistent store at
  `vector_db/chroma_store`, with a collection (default name
  `education_kb`) whose documents carry at least this metadata:
  ```json
  {
    "subject": "Math",
    "lesson_id": "5",
    "lesson_name": "Comparing Quantities",
    "grade": "KG1",
    "chapter": "Unit 1",
    "difficulty": "easy"
  }
  ```
- A Qwen API key (DashScope).

This module **never creates or re-embeds** the collection — it only
connects to what already exists.

---

## 3. Installation

```bash
# 1. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables
#    Edit secrets/.env and fill in your real Qwen API key and paths.
```

`secrets/.env`:
```
QWEN_API_KEY=your_qwen_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus

CHROMA_DB_PATH=./vector_db/chroma_store
COLLECTION_NAME=education_kb

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

LOG_LEVEL=INFO
STREAK_TO_LEVEL_UP=5
MISTAKES_TO_LEVEL_DOWN=3
```

---

## 4. Running the Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs (Swagger UI): `http://localhost:8000/docs`.

---

## 5. API Reference & Examples

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /start-session`
```bash
curl -X POST http://localhost:8000/start-session \
  -H "Content-Type: application/json" \
  -d '{"lesson_id": "5", "student_name": "Layla"}'
```
```json
{
  "session_id": "b3f1c9b2-...",
  "lesson_id": "5",
  "current_difficulty": "easy",
  "message": "Session started successfully."
}
```

### `POST /generate-question`
```bash
curl -X POST http://localhost:8000/generate-question \
  -H "Content-Type: application/json" \
  -d '{"session_id": "b3f1c9b2-..."}'
```
```json
{
  "session_id": "b3f1c9b2-...",
  "question": {
    "type": "mcq",
    "question": "What word do we use when one plate has more pieces than the other?",
    "choices": ["More", "Less", "Same quantity", "Zero"],
    "correct_answer": "More",
    "difficulty": "easy"
  }
}
```

If the retrieved lesson has no usable content, this endpoint returns
HTTP `422` with detail `"Insufficient lesson information."`.

### `POST /submit-answer`
```bash
curl -X POST http://localhost:8000/submit-answer \
  -H "Content-Type: application/json" \
  -d '{
        "session_id": "b3f1c9b2-...",
        "question": "What word do we use when one plate has more pieces than the other?",
        "correct_answer": "More",
        "player_answer": "More",
        "time_taken": 12
      }'
```
```json
{
  "is_correct": true,
  "score": 10,
  "difficulty": "easy",
  "time": 12,
  "feedback": "Great job, that's correct!"
}
```

### `POST /finish-session`
```bash
curl -X POST http://localhost:8000/finish-session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "b3f1c9b2-..."}'
```
```json
{
  "student_score": 84,
  "correct_answers": 17,
  "wrong_answers": 3,
  "accuracy": 85.0,
  "average_time": 14.2,
  "highest_level": "hard",
  "strengths": ["Answers quantity-comparison questions quickly and correctly"],
  "weaknesses": ["Struggles with hard-level pattern questions"],
  "common_mistakes": ["Most mistakes happened on hard difficulty questions"],
  "recommendations": ["Practice more pattern-rule exercises before the next session"],
  "question_analysis": [
    {"question": "...", "difficulty": "easy", "time": 12, "correct": true}
  ]
}
```

---

## 6. Design Notes

- **Deterministic correctness & stats.** The LLM never decides whether
  an answer is right or wrong, nor computes the numeric report fields
  — that logic lives in plain Python (`evaluator.py`, `report.py`).
  The LLM is only used for natural-language generation (questions,
  feedback text, and qualitative report sections), which keeps the
  system auditable and prevents hallucinated grading.
- **Strict lesson grounding.** The retriever filters ChromaDB strictly
  by `metadata["lesson_id"]`, and the question-generation prompt
  explicitly forbids the LLM from using outside knowledge, with a
  required `"Insufficient lesson information."` fallback.
- **In-memory sessions.** No SQL database is used, per spec. Sessions
  live in a process-wide dictionary inside `GameService`. For a
  multi-instance deployment, swap this for Redis without changing the
  public service API.
- **Adaptive difficulty.** A simple streak-based state machine:
  `STREAK_TO_LEVEL_UP` consecutive correct answers level the player up
  (capped at Hard); `MISTAKES_TO_LEVEL_DOWN` consecutive wrong answers
  level the player down (floored at Easy). Both thresholds are
  configurable via `.env`.
- **Prompt isolation.** All prompt text lives under `app/prompts/`,
  never inline in application logic, so prompt tuning never requires
  touching business logic.

---

## 7. Offline / Air-Gapped Environments

`retriever.py` loads `sentence-transformers/all-MiniLM-L6-v2` via
`HuggingFaceEmbeddings` to embed queries at retrieval time (the same
model used to build the existing collection). The first time this
model is used it is downloaded from Hugging Face and cached locally
under `~/.cache/huggingface`.

If you deploy in a network-restricted environment:
1. Pre-download the model once on a machine with internet access:
   ```python
   from sentence_transformers import SentenceTransformer
   SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
   ```
2. Copy the resulting `~/.cache/huggingface` folder to the target
   machine, or set `HF_HUB_OFFLINE=1` once it's cached, so no network
   call is attempted at runtime.

---

## 8. Error Handling Summary

| Scenario                                   | HTTP Status | Detail                                  |
|---------------------------------------------|:-----------:|------------------------------------------|
| Unknown `lesson_id`                          | 404         | Lesson not found                          |
| Unknown/expired `session_id`                 | 404         | Session not found                         |
| Lesson content insufficient for a question   | 422         | `Insufficient lesson information.`        |
| Chroma / LLM connectivity failure            | 502         | Upstream error detail                     |
| Any unhandled exception                      | 500         | Generic internal error (details in logs)  |
