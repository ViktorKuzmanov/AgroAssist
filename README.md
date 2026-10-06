# AgroAssist - Agricultural Recommendation Engine

AI-powered agricultural decision-support application for an agricultural pharmacy.

The system helps pharmacy employees make more informed, consistent, and situation-specific product recommendations by combining the pharmacy's product database, agricultural knowledge from an experienced agricultural engineer, customer/crop information, treatment history, and—when appropriate—information from external sources.

The application acts as a recommendation engine that helps answer questions such as:

> "What should we consider using for this particular crop and problem?"

The system evaluates the customer's specific situation and identifies relevant products from the pharmacy's own product database.

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



## Installation



### 1. Clone/Download the Repository

```bash
git clone <repository-url>
cd agricultural-recommendation-engine
```



### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```



## Goal

The long-term goal is to build an internal agricultural intelligence system that combines our family's agricultural knowledge with our pharmacy's product database and modern AI.

Instead of replacing the expertise of the people working in the pharmacy, the system should **make that expertise easier to access, more consistent, and easier to apply to each customer's specific situation.**

The final application should feel less like:

> "Ask an AI for agricultural advice."

and more like:

> **"Give the system the customer's situation and let it help us determine which of our products and treatment options we should consider."**

