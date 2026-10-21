"""
AgroAssist — FastAPI бекенд со RAG + read-only пристап до базата на производи.

За секое прашање:
1. Пребарува во базата ("Product") дали го имаме производот (read-only).
2. Го вчитува релевантното земјоделско знаење од agro_knowledge_base.txt (RAG).
3. Ги спојува двете во еден одговор.
"""

import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import APIConnectionError, APIStatusError, APITimeoutError, BadRequestError
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Gemini (бесплатен Google free tier) — РАСКОМЕНТИРАЈ оваа линија И соодветните
# редови подолу (модел + requirements.txt) ако сакаш да се врати Gemini како
# главен или резервен чат модел. Ниту една линија не е избришана.
# from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from db import db_available, format_products, search_products

load_dotenv()

# --- Чат модел (OpenAI / ChatGPT) --------------------------------------------

# Главен модел. gpt-5.4-mini е брз, евтин и прифаќа temperature=0 (детерминистички
# одговор — важно за правилото „наведи ги СИТЕ производи“).
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# Резервен модел — користи се САМО ако главниот падне (429 / 5xx / timeout).
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")

# Gemini — закоментиран засега (бесплатен, врати го кога сакаш да го полниш).
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# --- RAG: вчитај knowledge base и направи векторска база ---------------------

loader = TextLoader("./agro_knowledge_base.txt", encoding="utf-8")
documents = loader.load()

# Секој нумериран запис (култура/болест/штетник) е еден ред, па separator="\n"
# држи една цела ставка во еден фрагмент.
# TODO: nz dali  da namalam chunk_size pomalo da bide oti mnogu info prakja a nemam tolku info jas. 
# Primer za spanjakj ako prasa iam samo 500 chunk size mi treba a 1500 chunk size opfakja i drugite kulturi pod spanakjot 
# ama ne e strasno i 1500 da ostane, toa sepak zavisi i od bazata moze kje iam povekje u idnina
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=100,
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

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# --- Конструкција на чат моделите --------------------------------------------

# Некои модели од OpenAI (gpt-5, gpt-5-mini/nano/pro, сите gpt-6-*) одбиваат
# temperature != 1 со HTTP 400 ("Unsupported value: 'temperature'"). За нив
# temperature воопшто не се праќа. gpt-4o*, gpt-4.1*, gpt-5.2 и gpt-5.4* го
# прифаќаат temperature=0.
def _model_rejects_temperature(name: str) -> bool:
    return name == "gpt-5" or name.startswith("gpt-5-") or name.startswith("gpt-6")


def _chat_openai(name: str, temperature: bool = True) -> ChatOpenAI:
    """ChatOpenAI со автоматски избор дали смее да се прати temperature."""
    if not temperature or _model_rejects_temperature(name):
        return ChatOpenAI(model=name)
    # LangChain-овиот validator го чисти temperature за СИТЕ модели со "gpt-5" во
    # името, но суровиот API го прифаќа за gpt-5.2 и gpt-5.4 — затоа го ставаме
    # temperature ПО конструкцијата, кога validator-от веќе е поминал.
    m = ChatOpenAI(model=name)
    m.temperature = 0
    return m


model = _chat_openai(OPENAI_MODEL)
fallback_model = _chat_openai(OPENAI_FALLBACK_MODEL)

# Gemini (бесплатен) — закоментиран. Раскоментирај со import-от најгоре.
# model = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)

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


def _invoke_fallback(prompt, payload, primary_error):
    """Ако главниот модел падне (429 / 5xx / timeout) — одговара резервниот модел.

    Го користи OPENAI_FALLBACK_MODEL (по дифолт gpt-4o-mini). Ако и тој падне,
    клиентот добива 502 со читлив текст наместо 500 од непозната грешка.
    """
    print(
        f"[fallback] {OPENAI_MODEL} падна ({type(primary_error).__name__}) "
        f"→ пробувам {OPENAI_FALLBACK_MODEL}",
        flush=True,
    )
    try:
        chain = prompt | fallback_model | StrOutputParser()
        return chain.invoke(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Двата модела паднаа ({OPENAI_MODEL}, {OPENAI_FALLBACK_MODEL}): {exc}",
        ) from exc


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
    return {
        "status": "ok",
        "model": OPENAI_MODEL,
        "fallback_model": OPENAI_FALLBACK_MODEL,
        "database": db_available(),
    }

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

    # 2) Релевантно земјоделско знаење од agro_knowledge_base.txt.
    context = format_docs(retriever.invoke(question))

    prompt = ChatPromptTemplate.from_template(
        TEMPLATES.get(req.lang, TEMPLATES["en"])
    )
    chain = prompt | model | StrOutputParser()

    # Го печатам целиот промпт што оди кон моделот, за да се види во терминалот.
    print("\n" + "=" * 78)
    print(f"ПРОМПТ КОН {OPENAI_MODEL} ({req.lang})")
    print("=" * 78)
    print(prompt.format(products=products_ctx, context=context, question=question))
    print("=" * 78 + "\n", flush=True)

    payload = {"products": products_ctx, "context": context, "question": question}

    # TODO Ne go razbiram bas via del razgledaj go i razberi go
    try:
        reply = chain.invoke(payload)
    except BadRequestError as exc:
        # 400 што не е за temperature = вистинска грешка во барањето → не ретрајаме.
        if "temperature" not in str(exc).lower():
            raise HTTPException(status_code=502, detail=f"OpenAI грешка: {exc}") from exc
        # Моделот одби temperature — пробуваме уште еднаш, без него.
        print("[chat] моделот одби temperature — ретрај без него", flush=True)
        chain = prompt | _chat_openai(OPENAI_MODEL, temperature=False) | StrOutputParser()
        reply = chain.invoke(payload)
    except (APITimeoutError, APIConnectionError) as exc:
        reply = _invoke_fallback(prompt, payload, exc)
    except APIStatusError as exc:
        # 429 (rate limit) и 5xx се привремени → резервен модел.
        if exc.status_code < 429:
            raise HTTPException(status_code=502, detail=f"OpenAI грешка: {exc}") from exc
        reply = _invoke_fallback(prompt, payload, exc)

    return ChatResponse(reply=reply)


if __name__ == "__main__":
    # Локално/на Render: сервисот мора да слуша на 0.0.0.0 и на портата од $PORT.
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
