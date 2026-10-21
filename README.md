# AgroAssist - Agricultural Recommendation Engine

AI-powered agricultural decision-support application for an agricultural pharmacy.

The system helps pharmacy employees make more informed, consistent, and situation-specific product recommendations by combining the pharmacy's product database, agricultural knowledge from an experienced agricultural engineer, customer/crop information, treatment history, and—when appropriate—information from external sources.

The application acts as a recommendation engine that helps answer questions such as:

> "What should we consider using for this particular crop and problem?"

The system evaluates the customer's specific situation and identifies relevant products from the pharmacy's own product database.


## Installation


### 1. Clone/Download the Repository

```bash
git clone <repository-url>
cd agricultural-recommendation-engine
```

### 2. Install Backend Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure the OpenAI API Key

```bash
cp .env.example .env
# отвори .env и внеси го твојот OPENAI_API_KEY
```

> ⚠️ Клучот никогаш не се commit-ува. `.env` е во `.gitignore`.

### 4. Run the Chatbot Backend

```bash
uvicorn main:app --reload --port 8000
```

Тест:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Кога да прскам лозје против пламеница?"}],"lang":"mk"}'
```

Endpoint-и:

- `GET /health` — проверка дека серверот работи.
- `POST /chat` — прима `{ "messages": [{ "role": "user" | "assistant", "content": "..." }], "lang": "mk" | "en" }` и враќа `{ "reply": "..." }`.
- `GET /products/search?q=<барање>` — read-only пребарување во базата на производи (за тестирање).


## Deploy (Render — бесплатно)

**Жив сервис: https://agroassist-vxfe.onrender.com**

`render.yaml` го опишува сервисот, па деплојот е еден клик.

1. **render.com → New → Blueprint** → избери го репозиторијумот `ViktorKuzmanov/AgroAssist`.
   (Или **New → Web Service** со рачните вредности: Build `pip install -r requirements.txt`,
   Start `uvicorn main:app --host 0.0.0.0 --port $PORT`.)
2. Внеси ги env променливите (во Blueprint-от се бараат како `sync: false`):
   - `GEMINI_API_KEY` — https://aistudio.google.com/apikey (чат модел, бесплатен free tier)
   - `OPENAI_API_KEY` — потребен и при стартување (не само за embeddings): `OpenAIEmbeddings()`
     се создава при import, па без клуч сервисот не се буди
   - `DATABASE_URL` — PostgreSQL/Neon (read-only, база со производи)
   - `GEMINI_MODEL` — `gemini-3.1-flash-lite`
   - `ALLOWED_ORIGINS` — `*` за почеток; откако сајтот ќе биде жив, стави ги неговите адреси
   (на пример `https://viktoria-kristal.vercel.app,https://viktoriakristal.lovable.app`) и
   Render автоматски ќе редеплоира.
3. Сајтот го повикува овој сервис преку `VITE_ASSISTANT_URL` — таа вредност е запишана во
   `.env` на сајтот (`ViktoriaKristal-Website`), па бидејќи `VITE_*` се читаат **при build**,
   таа мора да биде поставена пред build-от (Vercel: Project → Settings → Environment
   Variables → Production + Preview).

Проверка:

```bash
curl https://agroassist-vxfe.onrender.com/health
# {"status":"ok","model":"gemini-3.1-flash-lite","database":true}
```

> ℹ️ Верзиите во `requirements.txt` се **пинирани** (chromadb 1.5.9, langchain 1.4.6 …) за да ја
> совпаѓаат шемата на верзионираната `chroma_db/`. Не ги отпуштај (`>=`) без да ја ребилдираш
> базата.

> ⚠️ Бесплатниот план **спие** по 15 минути неактивност — првата порака потоа чека 30–60 секунди
> додека сервисот се буди. Решение: бесплатен пингер (cron-job.org или UptimeRobot) на
> `https://<service>.onrender.com/health` на 10 минути.

> ℹ️ `chroma_db/` е верзионирана намерно: во спротивно секој restart ја гради векторската база
> одново, што значи платени OpenAI embedding повици и бавно стартување.

> ℹ️ При локално стартување Chroma ја допира `chroma_db/`, па `git status` покажува изменети
> бинарни фајлови. Отфрли ги пред commit:
>
> ```bash
> git checkout -- chroma_db/
> ```
>
> Почесто решение — скриј ги промените од Git, а верзионираните фајлови остануваат во репото
> (важи и за Render, кој ја clone-а базата при build):
>
> ```bash
> git update-index --skip-worktree chroma_db/
> # откажи:  git update-index --no-skip-worktree chroma_db/
> ```


## Поврзување со базата на производи (PostgreSQL/Neon)

Чатботот **само чита** од табелата `"Product"`. Ако во `.env` има `DATABASE_URL`, тој прво проверува дали го имаме производот и му ги дава достапните податоци (цена, категорија, активна супстанца, производител, залиха, рок, препорачана доза, каренца). Ако производот не се најде во базата, се користи постојната RAG логика (agro_knowledge_base.txt).

Конекцијата е read-only: `set_session(readonly=True)`, се користат само `SELECT` и нема промени во базата. Спојката се наоѓа во `db.py`. `DATABASE_URL` веќе е внесен во `AgroAssist/.env` (gitignored).

## How It Works

A pharmacy employee can enter information such as:

- Crop/culture — grapevine, onion, tomato, pepper, apple, etc.
- Disease or suspected problem
- Observed symptoms
- Growth/development stage
- Area size
- Previous treatments
- Previous products applied
- Treatment dates
- Other relevant information about the crop

The recommendation engine then combines multiple sources of information:

1. **Pharmacy Product Database**
  - Products available in our pharmacy
  - Crops the product can be used on
  - Diseases/problems it targets
  - Active ingredients
  - Product type
  - Dosage/application information
  - Application restrictions and other relevant information
2. **Agricultural Knowledge Base**
  - Knowledge and experience provided by our agricultural engineer
  - Crop-specific knowledge
  - Disease and pest knowledge
  - Treatment strategies
  - Application considerations
  - Practical recommendations and rules
3. **Customer & Treatment History**
  - Crop
  - Disease/problem
  - Growth stage
  - Previous applications
  - Application dates
  - Treatment history
  - Other relevant customer information
4. **External Information**
  - Internet research when additional or current information is required
  - Agricultural resources
  - Product documentation
  - Other trusted external sources

The system uses these sources to identify potentially suitable products from our own inventory and present the relevant information to the pharmacy employee.

## Architecture

The application is designed as a **decision-support system**, with the LLM acting as an intelligent reasoning and orchestration layer rather than being the source of truth for product information.

A simplified architecture is:

```text
                    Customer Information
                           |
                           v
                  +-------------------+
                  | Recommendation    |
                  | Engine / Agent    |
                  +---------+---------+
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
  Product Database   Agricultural KB     External Sources
          |                 |                  |
          +-----------------+------------------+
                            |
                            v
                   Candidate Products
                            |
                            v
                    Filtering / Ranking
                            |
                            v
                   AI Reasoning Layer
                            |
                            v
                  Employee Recommendation
```



## Recommendation Strategy

The system should not rely entirely on vector search or an LLM to determine which product to recommend.

Instead, the recommendation process should combine **structured filtering, retrieval, rules, and LLM reasoning**.

For example:

```text
Customer Problem
       |
       v
Identify Crop + Disease + Context
       |
       v
Filter Product Database
       |
       v
Find Potentially Compatible Products
       |
       v
Retrieve Relevant Agricultural Knowledge
       |
       v
Check Treatment History
       |
       v
Apply Rules / Constraints
       |
       v
Rank Candidates
       |
       v
LLM Generates Explanation
       |
       v
Employee Reviews Recommendation
```

This approach makes the system more reliable than simply putting the entire knowledge base into a vector database and asking an LLM what to recommend.

## RAG / Knowledge Retrieval

The agricultural knowledge base can use a RAG architecture where appropriate.

Knowledge can be organized into documents such as:

```text
knowledge/
├── crops/
│   ├── grapevine.md
│   ├── tomato.md
│   ├── onion.md
│   └── pepper.md
│
├── diseases/
│   ├── downy_mildew.md
│   ├── powdery_mildew.md
│   └── botrytis.md
│
├── pests/
│   └── ...
│
└── treatment-strategies/
    └── ...
```

However, structured information such as products, dosage, crops, active ingredients, and application information should remain in a **structured database** rather than being stored only as documents in a vector database.

The vector database should primarily be used for retrieving unstructured agricultural knowledge where semantic search is useful.

## Recommendation Ranking

Multiple products may be technically relevant.

The recommendation engine can therefore assign each candidate a relevance score based on factors such as:

```text
Crop compatibility
        +
Disease/problem compatibility
        +
Growth stage compatibility
        +
Active ingredient considerations
        +
Previous treatment history
        +
Agricultural knowledge
        +
Other application constraints
```

The result could look conceptually like:

```text
1. Product A
   High relevance

   Why:
   - Suitable for grapevines
   - Relevant to downy mildew
   - Compatible with the current growth stage
   - No problematic overlap with recent treatment

2. Product B
   Medium relevance

   Why:
   - Suitable for grapevines
   - Relevant to the disease
   - Additional considerations apply

3. Product C
   Low relevance

   Why:
   - Can be used on grapevines
   - Less suitable for the current situation
```

The application should show **why** a product was selected instead of only displaying a list of products.

## Role of the LLM

The LLM should primarily be responsible for:

- Understanding the employee's natural-language input
- Extracting structured information from the conversation
- Identifying missing information
- Querying the appropriate data sources
- Reasoning over retrieved information
- Comparing candidate products
- Explaining recommendations
- Asking useful follow-up questions
- Summarizing agricultural knowledge

The LLM should **not** be trusted as the authoritative source for:

- Product availability
- Product dosage
- Product composition
- Product registration
- Application restrictions
- Other pharmacy-specific product information

Those values should come from the application's trusted data sources.

## Technology Stack & Dependencies

- **Python** — Backend and recommendation engine
- **FastAPI** — API/backend framework
- **PostgreSQL** — Product and customer/treatment data
- **Vector Database** — Semantic retrieval of agricultural knowledge
- **LLM** — Natural-language understanding, reasoning, and orchestration
- **RAG** — Retrieval of relevant agricultural knowledge
- **LangChain / LangGraph** — AI workflow and agent orchestration
- **React** — Frontend application
- **Web Search API** — External agricultural information when required
- **Git / GitHub** — Version control and development workflow





## Goal

The long-term goal is to build an internal agricultural intelligence system that combines our family's agricultural knowledge with our pharmacy's product database and modern AI.

Instead of replacing the expertise of the people working in the pharmacy, the system should **make that expertise easier to access, more consistent, and easier to apply to each customer's specific situation.**

The final application should feel less like:

> "Ask an AI for agricultural advice."

and more like:

> **"Give the system the customer's situation and let it help us determine which of our products and treatment options we should consider."**

