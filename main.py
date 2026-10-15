"""
AgroAssist — минимален FastAPI бекенд поврзан со OpenAI.

Засега: едноставен chatbot endpoint (/chat) што ги праќа пораките до OpenAI.
Подоцна: RAG, embeddings, база на производи, правила итн.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

load_dotenv()

# --- Конфигурација ----------------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY не е поставен. Копирај .env.example во .env и внеси го клучот."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPTS = {
    "mk": (
        "Ти си AI асистент на Викторија Кристал, аграрна аптека. "
        "Одговарај кратко, јасно и корисно на прашања за прскање, ѓубрење, "
        "заштита на култури, болести и штетници. Ако не си сигурен, кажи дека "
        "не си сигурен и предложи да се консултира стручен агроном. "
        "Одговарај на македонски јазик."
    ),
    "en": (
        "You are the AI assistant of Viktoria Kristal, an agricultural pharmacy. "
        "Answer briefly, clearly and helpfully about spraying, fertilizing, crop "
        "protection, diseases and pests. If you are unsure, say so and suggest "
        "consulting a professional agronomist. Answer in English."
    ),
}


# --- Модели -----------------------------------------------------------------

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    lang: str = "mk"


class ChatResponse(BaseModel):
    reply: str


# --- Апликација -------------------------------------------------------------

app = FastAPI(title="AgroAssist Chatbot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    """Едноставна проверка дека серверот работи."""
    return {"status": "ok", "model": OPENAI_MODEL}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Прима историја на разговорот и враќа одговор од OpenAI."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Нема пораки.")

    system = SYSTEM_PROMPTS.get(req.lang, SYSTEM_PROMPTS["en"])

    # Ограничи ја историјата за да не расте неограничено.
    history = [m.model_dump() for m in req.messages[-20:]]

    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, *history],
        )
    except OpenAIError as exc:
        raise HTTPException(
            status_code=502, detail=f"Грешка од OpenAI: {exc}"
        ) from exc

    reply = completion.choices[0].message.content or ""
    return ChatResponse(reply=reply)
