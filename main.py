"""
AgroAssist — минимален FastAPI бекенд со едноставен RAG.

Го вчитува knowledgeBase.txt, го дели на chunks, прави embeddings и
одговара на прашања користејќи ги најрелевантните делови.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# --- RAG: вчитај knowledge base и направи векторска база ---------------------

loader = TextLoader("./knowledgeBase.txt", encoding="utf-8")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1200,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(documents)

vector_db = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vector_db.as_retriever(search_kwargs={"k": 4})

model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

TEMPLATES = {
    "mk": (
        "Ти си AI асистент на Викторија Кристал, аграрна аптека. Одговори кратко "
        "на прашањето користејќи го само следниот контекст:\n\n{context}\n\n"
        "Прашање: {question}"
    ),
    "en": (
        "You are the AI assistant of Viktoria Kristal, an agricultural pharmacy. "
        "Answer briefly using only the following context:\n\n{context}\n\n"
        "Question: {question}"
    ),
}


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


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
    """Најди релевантни делови од knowledge base-от и одговори."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Нема пораки.")

    question = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )

    context = format_docs(retriever.invoke(question))

    prompt = ChatPromptTemplate.from_template(
        TEMPLATES.get(req.lang, TEMPLATES["en"])
    )
    chain = prompt | model | StrOutputParser()

    reply = chain.invoke({"context": context, "question": question})
    return ChatResponse(reply=reply)
