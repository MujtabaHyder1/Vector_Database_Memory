# Vector Database Memory — TechCore Multi-Agent Support System

**Program**: Employability Life — Agentic AI & RAG (Week 4)  
**Date**: September 2026  
**Status**: ✅ Production-Ready • LangGraph Multi-Agent Orchestrated • Voice-Enabled

---

## 🌟 Overview & Key Enhancements

The **TechCore Support System** is an enterprise-grade AI IT support solution. Originally a standalone RAG chatbot, it has been comprehensively upgraded into an autonomous **LangGraph Multi-Agent Architecture** with integrated **domain tools**, **self-healing failure handlers**, **real-time step loading**, and **two-way Voice Mode (STT & TTS)**.

### Core Capabilities:
- **LangGraph Multi-Agent Team**: 7 specialized agents (Triage, Policy, Hardware, Diagnostics, Tools Executor, Responder, Memory) + dedicated Failure Handler.
- **Persistent Vector Memory**: 4 ChromaDB collections (`tech_policy_rag`, `tech_tickets`, `tech_episodic_memory`, `tech_declarative_memory`) using 384-dimensional cosine embeddings (`all-MiniLM-L6-v2`).
- **Domain Tools Layer (`tools.py`)**: 8 sandboxed IT tools including Hardware Catalog, Network Diagnostics, SLA Calculator, Ticket Creation & Lookup, System Status, and Vector Memory search.
- **PDF & Document Ingestion**: Direct upload and automated semantic chunking of `.pdf`, `.txt`, and `.md` files into ChromaDB vector memory (`pypdf` integration).
- **In-Chat Automated Ticket Creation**: Multi-agent recognition of ticket logging intent directly from the chat input with automatic SLA evaluation.
- **Resilient Error & Failure Handling**: Automatic graph-level exception routing to `node_failure_handler`, preventing crashes and delivering actionable remediation.
- **Ollama Silent Auto-Starter**: Automatically starts the local Ollama background server if offline (`ensure_running`).
- **Real-Time Step Loading & Agent Traces**: Dynamic visual status displaying agent reasoning, tool invocations, and live active agent pills.
- **ChatGPT-Style Unified Bottom Bar**: Single sleek pill input bar with built-in microphone voice dictation (`SpeechRecognition`), file attachment parsing, and text-to-speech audio read-aloud.
- **Modernized UI/UX**: High-contrast dark theme with interactive Ticket Central, Memory Vault, and Tool Sandbox tabs.

---

## 🏗️ Multi-Agent Architecture

```
                                  ┌──────────────────────────┐
                                  │   User Message / Voice   │
                                  └────────────┬─────────────┘
                                               │
                                               ▼
                                  ┌──────────────────────────┐
                                  │   1. Triage & Routing    │
                                  └──────┬─────┬─────┬───────┘
                     ┌───────────────────┘     │     └───────────────────┐
                     ▼                         ▼                         ▼
         ┌───────────────────────┐ ┌───────────────────────┐ ┌───────────────────────┐
         │ 2. Policy Specialist  │ │ 3. Hardware Specialist│ │4. Diagnostics Special.│
         └───────────┬───────────┘ └───────────┬───────────┘ └───────────┬───────────┘
                     └───────────────────┐     │     ┌───────────────────┘
                                         ▼     ▼     ▼
                                  ┌──────────────────────────┐
                                  │ 5. Domain Tools Executor │
                                  │  - Hardware Catalog      │
                                  │  - Network Diagnostics   │
                                  │  - SLA Calculator        │
                                  │  - Ticket Creation       │
                                  │  - ChromaDB Search       │
                                  └────────────┬─────────────┘
                                               ▼
                                  ┌──────────────────────────┐
                                  │ 6. Responder Agent (LLM) │
                                  └────────────┬─────────────┘
                                               ▼
                                  ┌──────────────────────────┐
                                  │ 7. Memory Agent (Embed)  │
                                  └────────────┬─────────────┘
                                               ▼
                                  ┌──────────────────────────┐
                                  │   Streamlit Web / Voice  │
                                  └──────────────────────────┘
```

> **Detailed Architecture Guide**: See [`ARCHITECTURE.md`](ARCHITECTURE.md) for Mermaid flowcharts, state schema specs, and failure recovery routing.

---

## 🧰 Domain Tools Layer (`tools.py`)

| Tool Identifier | Purpose | Outputs / Actions |
|---|---|---|
| `hardware_catalog_tool` | Lookup approved laptops, monitors, accessories | Specs, USD cost tier, lead times, approval hierarchy |
| `network_diagnostics_tool`| Troubleshoot VPN, Wi-Fi, BSOD, permissions | Step-by-step diagnostic checklist & escalation team |
| `sla_calculator_tool` | Calculate response & resolution targets | Resolution window, assigned engineer tier, on-call trigger |
| `ticket_creation_tool` | Create & register IT support tickets in ChromaDB | Ticket ID (`TECH-YYYYMMDD-...`), SLA commitments, assigned tier |
| `ticket_lookup_tool` | Search and filter existing tickets | Ticket IDs, documents, metadata, status |
| `search_policy_tool` | Semantic search against ChromaDB | Cosine-ranked policy chunks with relevance scores |
| `system_status_tool` | Live/mock status of IT infrastructure | SSO, GlobalProtect, Microsoft 365, AWS, Jira |
| `memory_lookup_tool` | Search declarative facts and episodic history | User-taught rules, previous conversations |

---

## 📄 PDF & Document Vector Ingestion

The system supports permanent knowledge ingestion of enterprise documents:
1. **Sidebar Upload**: In **🧠 Teach Agent Knowledge**, upload any `.pdf`, `.txt`, or `.md` file.
2. **Page-Level Parsing**: Uses `pypdf` to extract text page-by-page, sanitize content, and divide into semantic chunks.
3. **ChromaDB Indexing**: Generates 384-dimensional dense vectors with `SentenceTransformer` (`all-MiniLM-L6-v2`) and persists directly into `tech_declarative_memory`.
4. **Instant Multi-Agent Retrieval**: All specialist agents immediately query the newly ingested document context when answering employee inquiries.
5. **Chat Attachments**: Attach documents directly in the chat bar (`+` icon) for immediate on-the-fly conversational analysis.

---

## 🛡️ Error & Failure Handling

The system is engineered for **resilience and high availability**:
1. **Tool-Level Isolation**: Every tool executes in a sandboxed `try-except` block, returning a structured `ToolResult` object.
2. **LangGraph Failure Handler**: If any agent node throws an exception, conditional edges route directly to `node_failure_handler`.
3. **LLM Dropout Resilience**: If Ollama is offline or times out (90s limit), the system automatically executes structured vector synthesis.
4. **Silent Ollama Auto-Start**: If Ollama is not running, the application automatically launches `ollama serve` in the background.
5. **Persistent Vector Protection**: Collections use idempotent slug generation, preventing database corruption upon restarts.

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10+
- (Optional) [Ollama](https://ollama.ai/) with `mistral` model. The app automatically attempts to spin up Ollama if installed, and falls back to deterministic vector synthesis if unavailable.

### 2. Environment Setup
```bash
git clone https://github.com/MujtabaHyder1/Vector_Database_Memory.git
cd Vector_Database_Memory

# Activate existing or create virtual environment
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

# Install dependencies (chromadb, langgraph, pypdf, etc.)
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```
Open **http://localhost:8501** in your web browser.

---

## 💻 Dashboard Tabs

1. **💬 Support Desk**: Interactive multi-agent chat, live reasoning execution trace, active agent badges, speech-to-text dictation, and text-to-speech voice read-aloud.
2. **🎫 Ticket Central**: Live searchable explorer of all tickets stored in ChromaDB, with fast-track ticket creation directly inside the tab.
3. **🧠 Memory & Knowledge Vault**: Direct inspector for Policy Vectors, User-Taught Rules, Uploaded Document Chunks, and Episodic Conversation history.
4. **🛠️ Diagnostic Tools Sandbox**: Interactive testbed to execute Hardware lookups, VPN Diagnostics, SLA Calculations, and System Status checks.
5. **📊 System Telemetry & Health**: Real-time status indicators for Ollama, LangGraph, and ChromaDB persistent collections.

---

## 📁 Repository Structure

```
Vector_Database_Memory/
├── agent_graph.py           # LangGraph Multi-Agent StateGraph & Failure Handler
├── tools.py                 # Sandboxed Domain Tools (8 tools: Catalog, Diag, SLA, Tickets, Status)
├── tech_agent.py            # TechDepartmentAgent, TechMemoryRAG, OllamaClient, Document Ingestion
├── streamlit_app.py         # Streamlit UI with Unified Bottom Bar, Voice Mode & PDF Upload
├── ARCHITECTURE.md          # Multi-Agent Architecture Documentation & Mermaid Diagrams
├── SRS.md                   # Software Requirements Specification (9 FRs, 5 NFRs)
├── COMPLETION_SUMMARY.md    # Detailed Implementation & Test Verification Summary
├── START_HERE.md            # Quick-start onboarding and submission guide
├── requirements.txt         # Project dependencies (chromadb, langgraph, pypdf, etc.)
├── knowledge_base/
│   └── tech_department_policies.md   # Ground-truth corporate IT policies
└── .tech_agent_memory/      # Persistent ChromaDB storage directory
```
