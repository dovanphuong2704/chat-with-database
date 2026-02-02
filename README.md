# Chat with your PostgreSQL database (local LLM, fully offline)

Build a chatbot that answers questions about a **PostgreSQL** database and returns results as **tables** and **charts**. The app runs **fully offline** with a local LLM (**GPT-OSS 20B**) served by **Ollama**, plus safety layers for SQL and Python execution.

Article: https://mljar.com/blog/chatbot-python-postgresql-local-llm/

## What you get

- A local, offline chatbot that turns natural language into SQL.
- Results rendered as tables and Altair charts.
- A reproducible PostgreSQL demo dataset (customers, products, orders).
- Safety guards for LLM-generated SQL and Python.

## Architecture

- **PostgreSQL**: demo sales database (customers, products, orders)
- **Ollama + GPT-OSS 20B**: local LLM to generate SQL and Python
- **Python**: orchestration + safety layers
- **Mercury**: chat UI (notebook -> web app)

## Repo layout

```
postgresql/
  chat-with-postgresql.ipynb   # main Mercury notebook app
  dbclient.py                  # safe DB access + schema summary
  safeish.py                   # restricted Python executor
  setup_sales_db.sh            # demo DB setup via Docker
  generate_sales_csv.py        # data generator
  csv/                         # generated CSV files
```

## Quick start

### 1) Set up the PostgreSQL demo database

The demo database lives in `postgresql/`. It uses Docker and creates a read-only user for safe querying.

```bash
cd postgresql
chmod +x setup_sales_db.sh
./setup_sales_db.sh
```

Optional scale:

```bash
N_CUSTOMERS=5000 N_ORDERS=200000 COUNTRY_MODE=uniform ./setup_sales_db.sh
```

See `postgresql/README.md` for schema and connection details.

### 2) Start the local LLM (Ollama)

```bash
ollama run gpt-oss:20b
```

The first run downloads the model.

### 3) Install Python dependencies

```bash
pip install mercury ollama pandas altair psycopg[binary]
```

### 4) Open the notebook

Run the notebook in `postgresql/chat-with-postgresql.ipynb`.

The notebook:
- Loads the DB schema
- Sends it to the LLM as context
- Exposes tools for SQL and chart generation
- Shows results in a chat UI

### 5) Serve as a web app (optional)

```bash
mercury
```

Mercury will serve all `*.ipynb` notebooks in the directory.

## Database schema (demo)

Three tables:

- `customers`: customer details
- `products`: product catalog
- `orders`: transaction data

Example business questions you can ask:

- Which products sell the most?
- How many orders came from each country?
- What is monthly revenue?

## Safety layers

LLM-generated SQL and Python are restricted:

- **SQL**
  - Read-only user
  - Blocks `DROP`, `INSERT`, `UPDATE`, `GRANT`, etc.
  - Query timeout (e.g., 5 seconds)
  - Automatic `LIMIT` (e.g., 1000 rows)
- **Python**
  - No imports
  - Blocks `eval`, `exec`, `open`, `__import__`, etc.
  - AST validation + size/complexity limits
  - Restricted built-ins and scoped variables only

This is **safe-ish**, not a hardened sandbox, but it dramatically reduces risk.

## Example flow

1. User asks a question in the chat UI.
2. LLM generates a SQL query.
3. The app executes the query and shows a table.
4. If needed, the LLM generates Python (Altair) to plot a chart.

## Links

- Article: https://mljar.com/blog/chatbot-python-postgresql-local-llm/
- Full code (PostgreSQL example): https://github.com/mljar/chat-with-your-database/tree/main/postgresql

## License

See `LICENSE`.
