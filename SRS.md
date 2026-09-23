# Software Requirements Specification (SRS)

## Tech Department Support Agent with Vector Database Memory

**Program**: Employability Life — Agentic AI & RAG (Week 4)  
**Date**: September 2026  
**Team**: Dune  
**Status**: ✅ Complete & Verified

---

## 1. Executive Summary

This system is a **stateful AI support agent** for company tech departments. It uses **persistent vector memory** (ChromaDB), **policy RAG**, and a **local LLM** (Ollama) to handle employee queries about devices, software access, and IT policies.

The agent:

- Retrieves relevant company policies via semantic search before answering.
- Saves every conversation as episodic memory so past interactions improve future answers.
- Accepts user-taught facts as declarative memory that persist permanently.
- Creates structured tickets for device requests and device issues.
- Classifies issue priority automatically based on keyword analysis.
- Falls back to showing retrieved context when the LLM is unavailable.

**Core innovation**: A four-collection vector memory system (policy, episodic, declarative, tickets) that makes the agent smarter with every interaction.

---

## 2. System Components

### 2.1 Component Inventory

| Component | File | Responsibility |
|---|---|---|
| `TechMultiAgentWorkflow` | `agent_graph.py` | LangGraph StateGraph multi-agent orchestrator with step callbacks (8 nodes, resilient failure routing) |
| `Domain Tools Layer` | `tools.py` | 8 isolated tools: Hardware Catalog, Network Diag, SLA Calculator, Ticket Creation, Software Compatibility, Escalation Matrix, Compliance Check, Bulk Ticket Analysis |
| `FailureHandler` | `agent_graph.py` | Resilient error recovery node intercepting system/tool exceptions |
| `OllamaClient` | `tech_agent.py` | Wraps Ollama REST API with background auto-starter (`ensure_running`) and deterministic fallback |
| `TechMemoryRAG` | `tech_agent.py` | Manages 4 ChromaDB collections, embedding, search, PDF ingestion, ticket creation |
| `TechDepartmentAgent` | `tech_agent.py` | Top-level agent interface connecting LangGraph workflow and ChromaDB |
| `SearchResult` | `tech_agent.py` | Dataclass for ranked retrieval results |
| Streamlit UI | `streamlit_app.py` | Modern dashboard, live step loading, unified ChatGPT bottom bar (voice dictation + PDF upload), multi-tab layout |
| Policy knowledge base | `knowledge_base/tech_department_policies.md` | Source document for policy RAG |

### 2.2 Data Structures

#### SearchResult

```python
@dataclass
class SearchResult:
    id: str              # Document/memory ID
    content: str         # Full text content
    source: str          # Origin (file path, PDF, or memory kind)
    kind: str            # policy | episodic | declarative | ticket
    similarity: float    # 1 - cosine distance (0.0 – 1.0)
    metadata: dict       # Timestamps, categories, payload
```

---

## 3. Functional Requirements

### FR-01: Policy RAG Retrieval

| Field | Detail |
|---|---|
| **Requirement** | System must ingest a markdown policy document and retrieve relevant sections by semantic similarity |
| **Implementation** | `TechMemoryRAG.ingest_policies()` splits markdown by `## ` headings, embeds each section, stores in `tech_policy_rag` collection |
| **Collections** | `tech_policy_rag` (cosine, HNSW) |
| **Idempotency** | Uses stable slug IDs — re-running does not duplicate entries |

**Acceptance criteria**:
- [x] Policy document split into sections
- [x] Each section embedded and stored
- [x] Semantic search returns relevant sections for natural language queries
- [x] Re-ingestion is idempotent

---

### FR-02: Episodic Memory (Past Conversations)

| Field | Detail |
|---|---|
| **Requirement** | Every user interaction must be automatically saved for future retrieval |
| **Implementation** | `TechMemoryRAG.remember_interaction(query, answer, category)` |
| **Collection** | `tech_episodic_memory` (cosine, HNSW) |
| **Content format** | `"User asked: {query}\nAgent answered: {answer[:800]}"` |
| **Metadata** | `kind`, `category`, `timestamp` |

**Stored categories**:
- `conversation` — regular chat interactions
- `command` — slash command interactions
- `device_request` — device request submissions
- `device_issue` — device issue submissions
- `memory_update` — declarative memory teach events

**Acceptance criteria**:
- [x] Every `answer()` call auto-saves to episodic memory
- [x] Every `submit_device_request()` auto-saves
- [x] Every `submit_device_issue()` auto-saves
- [x] Every `teach_fact()` auto-saves
- [x] Episodic count increases after each interaction
- [x] Past interactions are retrievable via semantic search

---

### FR-03: Declarative Memory & Enterprise Document / PDF Ingestion

| Field | Detail |
|---|---|
| **Requirement** | Users must be able to teach persistent facts and ingest enterprise policy documents/PDFs |
| **Implementation** | `TechMemoryRAG.remember_fact(fact, category, source)` & `TechMemoryRAG.ingest_document_bytes(file_bytes, filename)` |
| **PDF Processing** | `pypdf.PdfReader` extracts page-by-page text, segments into semantic chunks, and tags metadata |
| **Collection** | `tech_declarative_memory` (cosine, HNSW) |
| **Seed data** | Operating rules seeded at initialization (`seed_declarative_memory()`) |
| **Teaching methods** | Direct PDF upload via chat pill bar, sidebar form ("Teach Declarative Memory"), or `/remember` chat command |
| **Metadata** | `kind`, `category`, `source`, `page`, `chunk`, `timestamp` |

**Acceptance criteria**:
- [x] Users can upload PDF documents directly from the chat input or sidebar
- [x] `pypdf` extracts text cleanly per page and splits into retrievable chunks
- [x] Users can teach facts via the sidebar form or `/remember <fact>`
- [x] Ingested document chunks and taught facts persist across session restarts
- [x] Vector similarity accurately retrieves uploaded document facts in chat reasoning
- [x] System seed rules are loaded on first run without duplication

---

### FR-04: Ticket Creation & Management (In-Chat & Fast-Track)

| Field | Detail |
|---|---|
| **Requirement** | Structured tickets can be generated automatically in-chat or submitted via fast-track UI |
| **Implementation** | `ticket_creation_tool` (agentic tool call) & `TechMemoryRAG.create_ticket(ticket_type, details)` |
| **Collection** | `tech_tickets` (cosine, HNSW) |
| **ID format** | `TECH-{YYYYMMDD}-{6-char hex}` |
| **Metadata** | `kind`, `ticket_type`, `status`, `timestamp`, `payload` (JSON) |
| **SLA Calculation** | Automated SLA calculation via `sla_calculator_tool` (Critical: 1h, High: 4h, Normal: 24h) |

**Ticket modalities**:
1. **Agentic In-Chat Creation**: Agent identifies user request/issue, invokes `ticket_creation_tool`, computes SLA, stores ticket, and confirms in conversation.
2. **Ticket Central UI**: Fast-track forms in Streamlit Ticket Central tab for direct device requests and incident reports.

**Acceptance criteria**:
- [x] Agent can invoke `ticket_creation_tool` directly from conversational prompts
- [x] Fast-track UI forms create tickets with unique IDs
- [x] Tickets are searchable via semantic similarity
- [x] Ticket content includes all provided fields
- [x] Ticket creation triggers episodic memory save

---

### FR-05: Priority Classification

| Field | Detail |
|---|---|
| **Requirement** | Device issues must be auto-classified as critical, high, or normal priority |
| **Implementation** | `TechDepartmentAgent._classify_priority(details)` |
| **Method** | Keyword scan across all detail values |

**Priority rules**:

| Priority | Keywords |
|---|---|
| Critical | lost, stolen, security, cannot work, unable to work, production, executive |
| High | vpn, crash, meeting, camera, microphone, blocked, urgent |
| Normal | Default (no matching keywords) |

**Acceptance criteria**:
- [x] "VPN crashes during meetings" → High
- [x] "Laptop stolen" → Critical
- [x] "Keyboard space bar sticking" → Normal

---

### FR-06: LLM-Powered Answer Generation & Auto-Starter

| Field | Detail |
|---|---|
| **Requirement** | Generate policy-grounded answers using local LLM, auto-start Ollama if offline, and provide structured fallback |
| **Implementation** | `OllamaClient.ensure_running()`, `TechDepartmentAgent._generate_answer()`, `_fallback_answer()` |
| **LLM** | Ollama REST API (`/api/generate`) with model `mistral` |
| **Auto-Starter** | `ensure_running()` checks Ollama status and spawns background process `ollama serve` if offline |
| **System prompt** | "You are a company tech department support agent. Use the policy RAG context, declarative memory, and episodic memory below to answer the employee." |
| **Context** | Top 6 search results formatted as `[KIND | similarity] content` |
| **Fallback** | Deterministic structured fallback showing retrieved context and actionable resolution steps when LLM is unavailable |

**Acceptance criteria**:
- [x] Ollama auto-starts if local daemon is inactive
- [x] When Ollama is running, responses are LLM-generated with retrieved context
- [x] When Ollama is unavailable, deterministic structured fallback renders retrieved context
- [x] Request timeout is 90 seconds
- [x] Network errors are caught gracefully and routed to fallback

---

### FR-07: Chat Commands

| Command | Action | Returns |
|---|---|---|
| `/help` | Show command list | Static help text |
| `/stats` | Query collection counts | Policy, ticket, episodic, declarative counts |
| `/policy <query>` | Search policies only (no memory) | Fallback answer with policy results |
| `/memory <query>` | Search all collections | Fallback answer with all results |
| `/remember <fact>` | Save declarative memory | Confirmation with memory ID |
| `/request` | Redirect to sidebar form | Instruction message |
| `/issue` | Redirect to sidebar form | Instruction message |

**Acceptance criteria**:
- [x] All 7 commands are recognized
- [x] Unknown commands return error message
- [x] Command interactions are saved to episodic memory
- [x] `/stats` counts match actual collection sizes

---

### FR-08: Memory Search & Ranking

| Field | Detail |
|---|---|
| **Requirement** | Queries must return ranked results from all relevant collections |
| **Implementation** | `TechMemoryRAG.search(query, top_k, include_memory)` |
| **Ranking** | All results sorted by `similarity` (1 − cosine distance) descending |
| **Top-K per collection** | Policy: 4, Tickets: 2, Declarative: 3, Episodic: 3 |
| **include_memory flag** | When False, only searches policy and ticket collections |

**Acceptance criteria**:
- [x] Results from all 4 collections are merged and sorted
- [x] `include_memory=False` excludes episodic and declarative
- [x] Empty collections are handled gracefully (no errors)
- [x] `n_results` is capped at actual collection count

---

### FR-09: Memory Clearing

| Field | Detail |
|---|---|
| **Requirement** | Agent memory can be reset without losing policies |
| **Implementation** | `TechMemoryRAG.clear_memory()` |
| **Clears** | Episodic, declarative, and ticket collections |
| **Preserves** | Policy collection |
| **Re-seeds** | Declarative operating rules are re-added after clearing |

---

## 4. Non-Functional Requirements

### NFR-01: Performance

| Metric | Target | Actual |
|---|---|---|
| Memory retrieval (HNSW) | < 100 ms | ✅ < 50 ms typical |
| LLM response generation | 2–10 s | ✅ Depends on model + hardware |
| Policy ingestion (cold start) | < 10 s | ✅ ~5 s for 6 sections |
| Embedding generation | < 50 ms/text | ✅ ~20 ms/text |

---

### NFR-02: Persistence & Durability

| Aspect | Implementation |
|---|---|
| Storage backend | ChromaDB `PersistentClient` |
| Storage path | `.tech_agent_memory/` (project-local) |
| Survives restarts | ✅ All collections persisted to disk |
| Survives crashes | ✅ ChromaDB writes are durable |

---

### NFR-03: Reliability & Error Handling

| Scenario | Handling |
|---|---|
| Ollama not running | `is_available()` returns False → fallback answer |
| Ollama request timeout | `requests.RequestException` caught → fallback with error note |
| Empty collection query | Count check before query → returns empty list |
| Empty declarative memory input | `ValueError` raised |
| Policy file missing | `ingest_policies()` returns early |

---

### NFR-04: Scalability

| Aspect | Detail |
|---|---|
| Indexing | HNSW (O(log n) retrieval) |
| Memory growth | Append-only (episodic and tickets grow with use) |
| Collection isolation | 4 separate collections prevent cross-contamination |
| Embedding model | `all-MiniLM-L6-v2` — lightweight, runs on CPU |

---

### NFR-05: Code Quality

| Aspect | Status |
|---|---|
| Type hints | ✅ All functions and methods |
| Dataclasses | ✅ `SearchResult` |
| Docstrings | ✅ On public methods |
| Separation of concerns | ✅ RAG, LLM, Agent, UI in separate classes/files |
| No global state | ✅ All state in class instances |

---

## 5. Architecture

### 5.1 Multi-Agent & Four-Layer Memory Model

```
PERCEPTION    Employee chat message, voice input, PDF attachment, or form submission
    │
    ▼
ORCHESTRATION TechMultiAgentWorkflow (LangGraph StateGraph, 8 Nodes):
              Intent Router ──┬──► Policy Specialist (RAG)
                              ├──► Memory Specialist (Episodic / Declarative)
                              ├──► Diagnostics Specialist (Network / Hardware)
                              ├──► Ticket Manager (Creation / Status / SLA)
                              └──► Failure Handler (Resilient fallback on exceptions)
    │
    ▼
TOOLS LAYER   8 Domain Tools (tools.py):
              • hardware_catalog_tool       • network_diagnostic_tool
              • sla_calculator_tool         • ticket_creation_tool
              • software_compatibility_tool • escalation_matrix_tool
              • compliance_check_tool       • bulk_ticket_analysis_tool
    │
    ▼
MEMORY        4 ChromaDB Collections (.tech_agent_memory/):
              • tech_policy_rag        (policy sections from markdown)
              • tech_episodic_memory    (past conversations & interactions)
              • tech_declarative_memory (taught facts & ingested PDF document chunks)
              • tech_tickets            (requests & issue tickets with metadata)
    │
    ▼
ACTION        OllamaClient.generate() or deterministic structured fallback
              remember_interaction() — auto-save interaction to episodic memory
```

### 5.2 RAG & Agentic Synthesis Pipeline

```
Retrieve   →   TechMemoryRAG.search(query)
                Returns SearchResult[] sorted by cosine similarity across 4 collections

Tools      →   LangGraph nodes trigger specialized domain tools (e.g. ticket creation, SLA, diagnostics)
                Outputs combined into structured AgentState

Augment    →   Build prompt:
                "You are an enterprise Tech Department Multi-Agent Support Team.
                 Context: [top results with kind + similarity]
                 Domain Tool Results: [diagnostics/tickets/catalog]
                 Employee message: {query}"

Generate   →   OllamaClient.generate(prompt, temperature=0.2)
                (Ollama auto-started if offline; deterministic fallback on failure)
```

---

## 6. Use Case Scenarios

### UC-01: Employee Asks a Policy Question

**Actor**: Employee via chat or voice dictation  
**Precondition**: Policy document ingested  
**Flow**:
1. Employee asks: "What is the laptop budget?"
2. Agent searches `tech_policy_rag` → finds "Standard laptop budget is 2000 USD"
3. Agent searches `tech_episodic_memory` → finds similar past questions
4. Agent searches `tech_declarative_memory` → finds taught rules and ingested PDF policies
5. Agent builds context prompt with top results
6. Ollama generates: "The standard laptop budget is $2000. Requests above this require finance and department head approval. Standard requests are fulfilled within 5 business days after manager approval."
7. Interaction saved to episodic memory

---

### UC-02: Employee Submits a Device Request

**Actor**: Employee via sidebar form  
**Precondition**: Agent loaded  
**Flow**:
1. Employee fills Device Request form (name, department, manager, device type, work mode, reason, date)
2. Agent creates ticket `TECH-20260912-EDBD43`
3. Ticket stored in `tech_tickets` collection
4. Interaction saved to episodic memory
5. Success message shown: "Device request created as TECH-20260912-EDBD43"

---

### UC-03: Employee Reports a Device Issue

**Actor**: Employee via sidebar form  
**Precondition**: Agent loaded  
**Flow**:
1. Employee fills Device Issue form (name, device/asset tag, urgency, issue summary, troubleshooting)
2. Agent auto-classifies priority: "VPN crashes during meetings" → **High**
3. Ticket created as `TECH-20260912-BF291C`
4. Interaction saved to episodic memory
5. Success message: "Device issue logged as TECH-20260912-BF291C with high priority"

---

### UC-04: Admin Teaches the Agent a New Rule or Uploads PDF

**Actor**: Admin via sidebar, `/remember` command, or PDF upload  
**Precondition**: Agent loaded  
**Flow**:
1. Admin uploads an enterprise PDF or types: `/remember Always require manager approval for laptops over $1500`
2. Document chunks / facts stored in `tech_declarative_memory` with source and metadata
3. Interaction saved to episodic memory
4. Confirmation displayed with chunk / memory IDs
5. Future queries about laptop approval or PDF policies retrieve these facts immediately

---

### UC-05: Agent Falls Back When LLM Is Unavailable

**Actor**: Employee via chat  
**Precondition**: Ollama offline and unreachable  
**Flow**:
1. Employee asks: "Can I get a new monitor?"
2. Agent searches all 4 collections — retrieves relevant policy and memory
3. `OllamaClient.ensure_running()` attempts auto-start; if unavailable, triggers fallback
4. Agent returns deterministic structured response showing retrieved context, policies, and actionable steps
5. Employee receives accurate policy context even without active local LLM

---

## 7. Testing & Verification

### Test Matrix

| Test | Method | Status |
|---|---|---|
| App loads at localhost:8501 | Browser | ✅ |
| `/help` returns all 7 commands | Chat | ✅ |
| `/stats` returns correct counts | Chat | ✅ |
| Policy search returns relevant sections | Chat | ✅ |
| Episodic memory increments after interaction | `/stats` before/after | ✅ |
| Declarative memory teachable via sidebar | Sidebar form | ✅ |
| Declarative memory teachable via `/remember` | Chat | ✅ |
| PDF document ingestion and retrieval | Chat/Sidebar upload | ✅ |
| Taught facts retrievable via `/memory` | Chat | ✅ |
| Device request creates ticket | Sidebar form | ✅ |
| Device issue creates ticket with priority | Sidebar form | ✅ |
| In-chat automated ticket generation | Chat | ✅ |
| Tickets increment in `/stats` | `/stats` before/after | ✅ |
| Fallback works when Ollama is down | Disconnect Ollama | ✅ |
| LLM generates grounded answer | Connect Ollama | ✅ |
| Memory persists across restarts | Stop/start Streamlit | ✅ |

### Test Scripts & Verification

1. `tools.py` unit verification:
```bash
python tools.py
```
2. Interactive multi-agent verification:
```bash
streamlit run streamlit_app.py
```

---

## 8. Policy Knowledge Base

The policy document (`knowledge_base/tech_department_policies.md`) contains 6 sections:

| Section | Content |
|---|---|
| Device Request Policy | Requestable devices, approval flow, $2000 budget limit, 5-day SLA, 4-year refresh, contractor rules |
| Device Issue and Support Policy | Reporting requirements, priority tiers (critical/high/normal), SLAs, lost/stolen procedure, repair rules |
| Access and Software Policy | Installation requests, paid software approval, security review for sensitive data, admin access rules, shared account prohibition |
| Remote Work Equipment Policy | Eligible equipment, internet reimbursement note, remote support scope |
| Device Return Policy | Return requirements, manager responsibility, recovery process |
| Knowledge Response Rules | Answer formatting guidelines for the agent |

---

## 9. Known Limitations

| Limitation | Detail |
|---|---|
| No memory decay | Older memories have equal weight to recent ones |
| No user isolation | All users share the same memory space |
| Single-instance only | ChromaDB file lock prevents concurrent processes |
| Context window | LLM context limited by model's max tokens |
| No authentication | No user login or role-based access |

---

## 10. Future Enhancements & Implementation Status

1. **Multi-modal document & PDF ingestion** — ✅ **IMPLEMENTED** (integrated via `pypdf`, page-level chunking, and declarative memory persistence).
2. **In-Chat Automated Ticket Creation** — ✅ **IMPLEMENTED** (integrated via `ticket_creation_tool` and `sla_calculator_tool`).
3. **Voice dictation & audio input** — ✅ **IMPLEMENTED** (integrated in ChatGPT bottom pill bar with `SpeechRecognition`).
4. **Memory decay** — reduce weight of older episodic memories (planned).
5. **User isolation** — per-user or per-department memory partitions (planned).
6. **Ticket lifecycle management** — update status, close, reopen tickets (planned).
7. **Distributed cloud vector DB** — scale to cloud vector database (planned).
8. **Authentication & SSO** — role-based access control with enterprise SSO (planned).

---

## 11. Sign-Off

| Checkpoint | Status |
|---|---|
| Specification reviewed | ✅ |
| All functional requirements met | ✅ |
| All non-functional requirements met | ✅ |
| Testing plan executed | ✅ |
| Architecture sound | ✅ |
| Documentation complete | ✅ |

**Status: APPROVED ✅**
