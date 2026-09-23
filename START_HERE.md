# Tech Department Support Agent — Quickstart & Onboarding Guide

**Status**: ✅ COMPLETE AND VERIFIED  
**Date**: September 2026  
**Program**: Employability Life — Agentic AI & RAG (Week 4)  
**System**: LangGraph Multi-Agent Architecture with ChromaDB Vector Memory

---

## 📦 What is in this Repository?

This repository contains a production-grade, stateful AI tech department support system featuring an **8-Node LangGraph Multi-Agent Team**, **8 Isolated Domain Tools**, **4 Persistent ChromaDB Vector Collections**, **Enterprise Document / PDF Ingestion via PyPDF**, and a **Modern Streamlit Dashboard with a Unified ChatGPT-Style Input Bar (Voice Dictation + Drag-and-Drop PDF Upload)**.

### Core Modules

| Component | File | Description |
|---|---|---|
| **Multi-Agent Orchestrator** | [`agent_graph.py`](file:///e:/Projects/Vector_Database_Memory/agent_graph.py) | LangGraph StateGraph coordinator with 8 specialized agent nodes & resilient failure handling |
| **Domain Tools Layer** | [`tools.py`](file:///e:/Projects/Vector_Database_Memory/tools.py) | 8 enterprise IT tools: Hardware Catalog, Network Diag, SLA Calculator, Ticket Creation, Software Compatibility, Escalation Matrix, Compliance Check, Bulk Ticket Analysis |
| **RAG & Memory Engine** | [`tech_agent.py`](file:///e:/Projects/Vector_Database_Memory/tech_agent.py) | ChromaDB manager with 4 persistent collections, `pypdf` document chunking pipeline, and Ollama auto-starter daemon |
| **User Interface** | [`streamlit_app.py`](file:///e:/Projects/Vector_Database_Memory/streamlit_app.py) | Streamlit app with unified ChatGPT bottom pill bar (`st.chat_input` with audio dictation + PDF upload), Ticket Central, and Visual Graph |
| **Policy Knowledge Base** | [`knowledge_base/tech_department_policies.md`](file:///e:/Projects/Vector_Database_Memory/knowledge_base/tech_department_policies.md) | Standard enterprise IT policy base covering equipment, budgets, SLAs, and security |

### Documentation Suite

- [`README.md`](file:///e:/Projects/Vector_Database_Memory/README.md) — Comprehensive architecture overview, quickstart, and feature guide.
- [`ARCHITECTURE.md`](file:///e:/Projects/Vector_Database_Memory/ARCHITECTURE.md) — Deep architectural specification with Mermaid flowcharts and data contracts.
- [`SRS.md`](file:///e:/Projects/Vector_Database_Memory/SRS.md) — Full Software Requirements Specification (9 Functional + 5 Non-Functional requirements).
- [`COMPLETION_SUMMARY.md`](file:///e:/Projects/Vector_Database_Memory/COMPLETION_SUMMARY.md) — Deliverables summary and 12-point live verification report.

---

## 🚀 How to Run

### Step 1: Activate Virtual Environment
Open PowerShell in the project directory:
```powershell
.\.venv\Scripts\activate
```

### Step 2: Verify Dependencies (Optional)
All required dependencies (`langgraph`, `chromadb`, `sentence-transformers`, `pypdf`, `torchvision`, `SpeechRecognition`) are installed:
```powershell
pip install -r requirements.txt
```

### Step 3: Verify Domain Tools
Run the standalone verification suite for all 8 domain tools:
```powershell
python tools.py
```
*Expected: 8/8 domain tools execute successfully with structured outputs.*

### Step 4: Launch the Streamlit App
```powershell
streamlit run streamlit_app.py
```
Then navigate to:
👉 **`http://localhost:8501`**

> [!NOTE]
> **Silent Ollama Auto-Starter**: You do NOT need to manually launch Ollama in a separate terminal. The application's `OllamaClient.ensure_running()` method will automatically detect if the Ollama daemon is offline and spawn `ollama serve` in the background. If Ollama is not installed, the system cleanly triggers its deterministic structured fallback.

---

## 💡 Key Features & Interaction Modes

### 1. Unified ChatGPT-Style Bottom Pill Bar
Located at the bottom of the screen (matching ChatGPT's modern ergonomics):
- **Text Queries**: Type any tech support or policy inquiry.
- **Voice Dictation**: Click the microphone icon directly inside the input pill bar to speak your issue. The audio is transcribed via Google Speech Recognition.
- **Drag-and-Drop PDF Upload**: Click the attachment (+) button inside the input pill bar or sidebar to upload an enterprise policy PDF. The system automatically extracts text via `pypdf`, chunks it by page, computes vector embeddings, and stores it in `tech_declarative_memory`.

### 2. In-Chat Automated Ticket Generation
- Whenever an employee reports an issue (e.g. *"I lost my laptop"* or *"VPN is broken"*), the LangGraph team identifies the need for a ticket, calls `ticket_creation_tool`, computes real-time SLA with `sla_calculator_tool`, logs the ticket into ChromaDB, and returns the ticket ID (`TECH-{DATE}-{HEX}`) directly in the chat.

### 3. Ticket Central Tab
- Switch to the **Ticket Central** tab in the UI to view active tickets or submit fast-track device requests and incident reports.

### 4. Live Visual Graph & Agent Step Tracking
- Switch to the **Visual Graph** tab to inspect the real-time execution flow across the 8 LangGraph nodes.
- In the chat view, expand the live step pill (`st.status`) to view which tools were called and what context was retrieved.

### 5. Slash Commands
Type slash commands directly into the chat input bar:
- `/help` — List all available commands.
- `/stats` — Show live document counts across all 4 ChromaDB collections.
- `/policy <query>` — Search company policies directly.
- `/memory <query>` — Search episodic and declarative memories.
- `/remember <fact>` — Teach the agent a persistent operational rule.

---

## 📁 Repository Structure

```
Vector_Database_Memory/
├── agent_graph.py                    # LangGraph 8-node multi-agent orchestrator
├── tools.py                          # 8 isolated domain tools
├── tech_agent.py                     # Core agent, ChromaDB RAG, PDF ingestion, Ollama auto-starter
├── streamlit_app.py                  # Streamlit dashboard (voice, PDF upload, ticket central)
├── knowledge_base/
│   └── tech_department_policies.md   # Markdown policy document (6 sections)
├── .tech_agent_memory/               # ChromaDB persistent vector storage directory
├── ARCHITECTURE.md                   # Full architectural specification & Mermaid diagrams
├── SRS.md                            # Software Requirements Specification
├── README.md                         # Main repository guide
├── START_HERE.md                     # This quickstart onboarding document
├── COMPLETION_SUMMARY.md             # Test results & deliverables summary
├── requirements.txt                  # Python dependencies
└── runtime.txt                       # Python runtime specification
```

---

## 🎯 Verification Checklist

- [x] Streamlit web application running at `http://localhost:8501`
- [x] LangGraph 8-node multi-agent team (`agent_graph.py`)
- [x] 8 domain tools in [`tools.py`](file:///e:/Projects/Vector_Database_Memory/tools.py) with unit verification
- [x] 4 ChromaDB persistent vector collections (`tech_policy_rag`, `tech_episodic_memory`, `tech_declarative_memory`, `tech_tickets`)
- [x] Enterprise document & PDF ingestion pipeline (`pypdf`)
- [x] Unified ChatGPT-style bottom pill bar with audio dictation and file attachment
- [x] Ollama background auto-starter (`ensure_running`)
- [x] All documentation updated and synchronized across the repository
