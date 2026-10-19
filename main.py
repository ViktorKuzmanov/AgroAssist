"""
AgroAssist — FastAPI бекенд со RAG + read-only пристап до базата на производи.

За секое прашање:
1. Пребарува во базата ("Product") дали го имаме производот (read-only).
2. Го вчитува релевантното земјоделско знаење од knowledgeBase.txt (RAG).
3. Ги спојува двете во еден одговор.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from db import db_available, format_products, search_products

load_dotenv()

# OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# TODO moze Fallback во код da stavam: ако 3.8-flash hitne rate limit i фрли 429, автоматски ретрај со flash-lite 
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
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

# Кеширање: ако веќе има зачувана Chroma база на диске, ја вчитуваме директно
# и НЕ праќаме документи до OpenAI одново (штедиме пари). Се пресметува само
# ембедингот на новото прашање од корисникот.
CHROMA_PERSIST_DIR = "./chroma_db"
embedding = OpenAIEmbeddings()

if os.path.isdir(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
    vector_db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding,
    )
    print(f"[rag] вчитана постоечка Chroma база од {CHROMA_PERSIST_DIR} (без нови ембединги)")
else:
    vector_db = Chroma.from_documents(
        chunks,
        embedding=embedding,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[rag] изработена нова Chroma база и зачувана во {CHROMA_PERSIST_DIR}")

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# model = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

# TODO voa da gi vrakja site proizvodi nema da bide valjda zasekogas oti ne e praktino
#  ja kje imam 50 proizvodi za pepelnica ako prasa userot primer i nema site da gi gleda on
# Add prompt template
TEMPLATES = {
    "mk": (
        "Ти си AI асистент на Викторија Кристал, земјоделска аптека.\n\n"
        "ЈАЗИК: Одговарај на истиот јазик на кој е поставено прашањето — на "
        "македонски ако прашањето е на македонски, а на англиски ако прашањето е "
        "на англиски. Одговарај јасно и кратко, но секогаш наведи ги сите "
        "производи од списокот.\n\n"
        "ПРОИЗВОДИ НАЈДЕНИ ВО НАШАТА БАЗА (овие производи ПОСТОЈАТ и се достапни):\n"
        "{products}\n\n"
        "ЗЕМЈОДЕЛСКО ЗНАЕЊЕ:\n{context}\n\n"
        "Прашање: {question}\n\n"
        "Правила:\n"
        "- Наведи ГИ СИТЕ производи од списокот, без исклучок — дури и кога имаат "
        "само име и цена. Никогаш не изоставај производ што е во списокот.\n"
        "- Секој производ наведен во списокот ПОСТОИ во нашата база. Никогаш не "
        "кажувај дека не го најдовме ако името е во списокот.\n"
        "- Ако прашањето е за производ од списокот, кажи дека го имаме и наведи ги "
        "достапните податоци (цена, категорија, активна супстанца, производител, "
        "залиха, рок, препорачана доза, каренца).\n"
        "- Ако некој податок недостасува, кажи дека не е наведен. Не измислувај "
        "цени, залиха или состојки.\n"
        "- Само ако списокот е празен (NONE), кажи дека моментално не го најдовме "
        "производот и искористи го земјоделското знаење за да помогнеш."
    ),
    "en": (
        "You are the AI assistant of Viktoria Kristal, an agricultural pharmacy.\n\n"
        "LANGUAGE: Answer in the same language as the question — English if the "
        "question is in English, Macedonian if the question is in Macedonian. "
        "Answer clearly and briefly, but always list every product from the list.\n\n"
        "PRODUCTS FOUND IN OUR DATABASE (these products EXIST and are available):\n"
        "{products}\n\n"
        "AGRICULTURAL KNOWLEDGE:\n{context}\n\n"
        "Question: {question}\n\n"
        "Rules:\n"
        "- List EVERY product from the list above, without exception — even when it "
        "has only a name and a price. Never omit a product that is in the list.\n"
        "- Every product listed above EXISTS in our database. Never say it was not "
        "found if its name is in the list.\n"
        "- If the question is about a listed product, say we have it and give the "
        "available details (price, category, active substance, manufacturer, stock, "
        "deadline, default dosage, karenca).\n"
        "- If a detail is missing, say it is not listed. Never invent prices, stock, "
        "or ingredients.\n"
        "- Only if the list is empty (NONE), say we did not find the product and use "
        "the agricultural knowledge to help."
    ),
}

# spoj gi site docs t.e. rezultati od retriever=ot i spoj gi so prazen red. Ova kje bide context-ot
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
    return {"status": "ok", "model": GEMINI_MODEL, "database": db_available()}

# TODO: Ne se koriste nigde vaa func moze da se isbrise pokasno samo za testiranje e 
@app.get("/products/search")
def products_search(
    q: str = Query(..., min_length=1, description="Барање (име, супстанца, категорија)"),
    limit: int = Query(6, ge=1, le=25),
) -> dict:
    """Read-only преглед на тоа што чатботот го гледа во базата. За тестирање."""
    if not db_available():
        raise HTTPException(status_code=503, detail="Базата не е поврзана.")
    rows = search_products(q, limit=limit)
    return {"query": q, "count": len(rows), "products": rows}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Провери во базата дали го имаме производот, додади знаење и одговори."""
    if not req.messages:
        raise HTTPException(status_code=400, detail="Нема пораки.")

    question = next(
        (m.content for m in reversed(req.messages) if m.role == "user"), ""
    )

    # 1) Read-only пребарување во базата на производи.
    if db_available():
        products_ctx = format_products(search_products(question, limit=6))
    else:
        products_ctx = "Базата не е поврзана (нема DATABASE_URL)."

    # 2) Релевантно земјоделско знаење од knowledgeBase.txt.
    context = format_docs(retriever.invoke(question))

    prompt = ChatPromptTemplate.from_template(
        TEMPLATES.get(req.lang, TEMPLATES["en"])
    )
    chain = prompt | model | StrOutputParser()

    # Го печатам целиот промпт што оди кон моделот, за да се види во терминалот.
    print("\n" + "=" * 78)
    print(f"ПРОМПТ КОН {GEMINI_MODEL} ({req.lang})")
    print("=" * 78)
    print(prompt.format(products=products_ctx, context=context, question=question))
    print("=" * 78 + "\n", flush=True)

    reply = chain.invoke(
        {"products": products_ctx, "context": context, "question": question}
    )
    return ChatResponse(reply=reply)
