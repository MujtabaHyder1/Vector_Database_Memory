# Vector Database Memory — Tech Department Support Agent

**Program**: Employability Life — Agentic AI & RAG (Week 4)  
**Date**: September 2026  
**Team**: Dune  
**Status**: ✅ Production-Ready

---

## Overview

A **stateful AI support agent** for company tech departments that uses **persistent vector memory** (ChromaDB), **policy RAG**, and a **local LLM** (Ollama) to answer employee questions about devices, software, access, and IT policies.

Unlike a stateless chatbot, this agent:

- **Remembers every conversation** (episodic memory) and uses past interactions to improve future answers.
- **Stores taught facts and instructions** (declarative memory) that persist across sessions.
- **Retrieves relevant policy** sections via semantic search before generating an answer.
- **Creates and tracks tickets** for device requests and device issues.
- **Falls back gracefully** to retrieved context when the LLM is unavailable.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Streamlit UI                              │
│   Chat · Sidebar forms · /commands · Metrics                   │
└────────────────────────┬───────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │ TechDepartmentAgent │
              │   answer()          │
              │   submit_device_*() │
              │   teach_fact()      │
              └──────────┬──────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
  ┌───────▼────────┐          ┌────────▼────────┐
  │ TechMemoryRAG  │          │  OllamaClient   │
  │  search()      │          │  generate()     │
  │  remember_*()  │          │  is_available() │
  │  create_ticket()│          └────────────────┘
  └───────┬────────┘
          │
  ┌───────▼──────────────────────────────┐
  │           ChromaDB (Persistent)       │
  │                                       │
  │  tech_policy_rag        6 sections    │
  │  tech_episodic_memory   conversations │
  │  tech_declarative_memory facts/rules  │
  │  tech_tickets           requests/issues│
  └───────────────────────────────────────┘
          │
  ┌───────▼───────────────┐
  │  SentenceTransformer  │
  │  all-MiniLM-L6-v2     │
  │  384-dim embeddings   │
  └───────────────────────┘
```

### Memory Model

| Memory Type | Collection | Purpose | Example |
|---|---|---|---|
| **Policy RAG** | `tech_policy_rag` | Company policies split by markdown section | "Standard laptop budget is 2000 USD" |
| **Episodic** | `tech_episodic_memory` | Past conversations auto-saved after every interaction | "User asked about VPN crash — agent suggested reinstall" |
| **Declarative** | `tech_declarative_memory` | Facts and instructions taught by users or seeded at init | "Always require manager approval for laptops over $1500" |
| **Tickets** | `tech_tickets` | Device requests and device issues with structured fields | TECH-20260912-EDBD43 — Laptop request for John Smith |

All four collections use **cosine similarity** with **HNSW indexing** for O(log n) retrieval.

---

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Ollama | Latest | Local LLM server |
| mistral model | — | Default LLM (configurable) |

### 1. Clone and install

```bash
git clone https://github.com/MujtabaHyder1/Vector_Database_Memory.git
cd Vector_Database_Memory

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Start Ollama

```bash
# Terminal 1 — start the server
ollama serve

# Terminal 2 — pull the model
ollama pull mistral
```

> The agent works without Ollama too — it falls back to showing retrieved policy context directly.

### 3. Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### 4. (Optional) Run the test notebook

```bash
python -m nbconvert --to notebook --execute Tech_Department_Agent_Test.ipynb \
  --output Tech_Department_Agent_Test.executed.ipynb
```

---

## Features

### Chat Commands

| Command | Description |
|---|---|
| `/help` | Show all available commands |
| `/policy <query>` | Search tech policies only |
| `/memory <query>` | Search policies + tickets + episodic + declarative memory |
| `/remember <fact>` | Save a declarative memory for future answers |
| `/stats` | Show counts for all memory collections |
| `/request` | Redirect to the Device Request sidebar form |
| `/issue` | Redirect to the Device Issue sidebar form |

### Sidebar

- **Metrics panel** — live counts for policy sections, tickets, episodic memory, declarative memory.
- **Teach Declarative Memory** — type a fact or instruction and the agent will remember it permanently.
- **Device Request form** — structured form capturing employee, department, manager, device type, work mode, business reason, preferred date.
- **Device Issue form** — structured form capturing employee, device/asset tag, urgency, issue summary, troubleshooting tried.

### Answer Generation Flow

```
User message
    │
    ├── Is it a /command? ──► Handle command directly
    │
    └── Regular question
            │
            ▼
        ┌───────────────────────────┐
        │  TechMemoryRAG.search()   │
        │  Query all 4 collections  │
        │  Sort by cosine similarity│
        └───────────┬───────────────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Build prompt with top 6  │
        │  retrieved context chunks │
        └───────────┬───────────────┘
                    │
            ┌───────┴───────┐
            │               │
     Ollama available?   Not available
            │               │
            ▼               ▼
     LLM generates     Fallback: show
     grounded answer    retrieved context
            │               │
            └───────┬───────┘
                    │
                    ▼
        ┌───────────────────────────┐
        │  Save to episodic memory  │
        │  (auto, every interaction)│
        └───────────────────────────┘
```

### Ticket Priority Classification

Device issues are auto-classified by scanning the issue text:

| Priority | Trigger keywords |
|---|---|
| **Critical** | lost, stolen, security, cannot work, unable to work, production, executive |
| **High** | vpn, crash, meeting, camera, microphone, blocked, urgent |
| **Normal** | Everything else |

---

## File Structure

```
Vector_Database_Memory/
├── streamlit_app.py                  Streamlit UI (chat + sidebar forms)
├── tech_agent.py                     Core agent: OllamaClient, TechMemoryRAG,
│                                       TechDepartmentAgent (487 lines)
├── knowledge_base/
│   └── tech_department_policies.md   Policy document (6 sections, 70 lines)
├── .tech_agent_memory/               ChromaDB persistent storage (auto-created)
├── Vector_Database_Memory.ipynb      Original Week 4 notebook (HR/IT agents)
├── Tech_Department_Agent_Test.ipynb  Test notebook for tech agent
├── requirements.txt                  Python dependencies
├── runtime.txt                       Python version spec
├── README.md                         This file
├── SRS.md                            Software Requirements Specification
├── COMPLETION_SUMMARY.md             Verification and test results
├── START_HERE.md                     Legacy quick-start guide
└── .gitignore                        Git exclusions
```

---

## Technical Specifications

### Embeddings & Vector Store

| Parameter | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` (SentenceTransformers) |
| Dimensions | 384 |
| Distance metric | Cosine |
| Indexing | HNSW |
| Backend | ChromaDB `PersistentClient` |
| Storage path | `.tech_agent_memory/` (project-local) |

### LLM

| Parameter | Value |
|---|---|
| Provider | Ollama (local) |
| Default model | `mistral` |
| API endpoint | `http://localhost:11434/api/generate` |
| Temperature | 0.2 |
| Timeout | 90 seconds |
| Fallback | Policy-context response when Ollama is unavailable |

### Performance

| Operation | Latency |
|---|---|
| Memory retrieval (HNSW) | < 100 ms |
| LLM response generation | 2–10 s (depends on model + hardware) |
| Policy ingestion (first run) | ~5 s (6 sections, embedding + insert) |

---

## How It Connects to Previous Weeks

| Week | Concept | How it's used here |
|---|---|---|
| **Week 2** | Vector embeddings & semantic search | `all-MiniLM-L6-v2` embeds queries and documents; ChromaDB stores and retrieves by cosine similarity |
| **Week 3** | RAG pipelines | Retrieve policy + memory → Augment the prompt → Generate answer via Ollama |
| **Week 4** | Stateful agents with persistent memory | Episodic + declarative memory persists across sessions; agent learns from every interaction |

---

## Dependencies

```
chromadb
sentence-transformers
requests
streamlit
numpy
nbconvert
```

All are listed in `requirements.txt`.

---

## Troubleshooting

### Ollama not connecting

```
Error: Cannot connect to Ollama at http://localhost:11434
```

1. Start the server: `ollama serve`
2. Pull the model: `ollama pull mistral`
3. Verify: `curl http://localhost:11434/api/tags`

### Model not found

```
Error: Model 'mistral' not found
```

Install it: `ollama pull mistral`  
Or switch to another model in the Streamlit sidebar.

### ChromaDB lock error

```
Error: Cannot write to .tech_agent_memory/
```

Ensure only one process accesses the database at a time. Stop any running notebook or Streamlit instance before restarting.

### Out of memory (LLM)

Try a smaller model:
```bash
ollama pull orca-mini
```
Then change the model name in the Streamlit sidebar.

---

## License

This project was developed as part of the **Employability Life — Agentic AI & RAG** program (Week 4).
