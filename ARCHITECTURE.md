# TechCore Multi-Agent Support System Architecture

## 1. High-Level Architectural Overview

The **TechCore Support System** is an enterprise-grade AI IT support solution. It orchestrates a team of specialized agents using **LangGraph**, integrates a persistent 4-collection vector memory using **ChromaDB**, executes a sandboxed suite of **IT Domain Tools**, features an **automated Failure & Error Recovery Boundary**, and provides a modern web interface with **Voice Mode (STT & TTS)** via **Streamlit**.

```mermaid
graph TD
    User([User / Employee]) <-->|Chat / Voice Mode| UI[Streamlit Modern Web Dashboard]
    
    subgraph UI_Layer [Frontend & Voice Interaction]
        UI -->|Query / Form Submit| LangGraph[LangGraph Multi-Agent Engine]
        UI -->|Speech-to-Text| STT[Web Speech Recognition API]
        UI <--|Text-to-Speech| TTS[Web Speech Synthesis Audio]
    end

    subgraph MultiAgent_Layer [LangGraph Multi-Agent StateGraph]
        LangGraph --> Triage[1. Triage & Routing Agent]
        
        Triage -->|Policy Query| PolicyAgent[2. Policy & Compliance Specialist]
        Triage -->|Hardware / Order| HardwareAgent[3. Hardware & Procurement Specialist]
        Triage -->|Device Issue / Outage| DiagAgent[4. IT Diagnostics Specialist]
        
        PolicyAgent --> ToolsExec[5. Sandboxed Tools Execution Node]
        HardwareAgent --> ToolsExec
        DiagAgent --> ToolsExec
        
        ToolsExec --> Responder[6. Synthesis & Responder Agent]
        Responder --> MemoryAgent[7. Episodic & Declarative Memory Agent]
        
        %% Error Recovery Routing
        Triage -.->|Exception / Failure| FailureHandler[8. Resilient Failure & Recovery Handler]
        PolicyAgent -.->|Exception / Failure| FailureHandler
        HardwareAgent -.->|Exception / Failure| FailureHandler
        DiagAgent -.->|Exception / Failure| FailureHandler
        ToolsExec -.->|Exception / Failure| FailureHandler
        Responder -.->|Exception / Failure| FailureHandler
        
        FailureHandler --> MemoryAgent
        MemoryAgent --> EndNode([Workflow Complete])
    end

    subgraph Tools_Layer [Domain Tools Layer (tools.py)]
        ToolsExec <--> ToolPolicy[search_policy_tool]
        ToolsExec <--> ToolTicket[ticket_lookup_tool]
        ToolsExec <--> ToolTicketCreate[ticket_creation_tool]
        ToolsExec <--> ToolHW[hardware_catalog_tool]
        ToolsExec <--> ToolNet[network_diagnostics_tool]
        ToolsExec <--> ToolSLA[sla_calculator_tool]
        ToolsExec <--> ToolMem[memory_lookup_tool]
        ToolsExec <--> ToolSys[system_status_tool]
    end

    subgraph Storage_Layer [Persistent Storage & LLM]
        ToolPolicy <--> ChromaPolicy[(ChromaDB: tech_policy_rag)]
        ToolTicket <--> ChromaTicket[(ChromaDB: tech_tickets)]
        ToolTicketCreate --> ChromaTicket
        MemoryAgent <--> ChromaEpisodic[(ChromaDB: tech_episodic_memory)]
        MemoryAgent <--> ChromaDecl[(ChromaDB: tech_declarative_memory)]
        Responder <--> OllamaLLM[Ollama Local LLM REST API]
    end
```

---

## 2. LangGraph Multi-Agent StateGraph Specification

The orchestration is implemented in `agent_graph.py` using `langgraph.graph.StateGraph` governed by the `AgentState` schema.

### State Schema (`AgentState`)
```python
class AgentState(TypedDict):
    user_message: str
    intent: str                  # "policy_query" | "hardware_request" | "device_issue" | "general"
    urgency: str                 # "critical" | "high" | "normal"
    extracted_entities: Dict[str, Any]
    active_agent: str            # Current executing agent name
    agent_trace: List[str]       # Sequential log of agent steps for UI visualizer
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    sources: List[Dict[str, Any]]
    final_answer: str
    error: Optional[str]         # Populated if any failure occurs
    status: str                  # "in_progress" | "completed" | "recovered" | "failed"
```

### Agent Roles & Responsibilities

| Agent Node | Core Responsibility | Downstream Target |
| :--- | :--- | :--- |
| **`triage_agent`** | Natural language parsing, intent categorization, entity extraction (device brand, budget), priority classification (critical/high/normal). | `policy_agent`, `hardware_agent`, `diagnostics_agent`, or `failure_handler` |
| **`policy_agent`** | Corporate compliance checks, reimbursement limits, acceptable use guidelines, software seat policies. | `tools_execution` |
| **`hardware_agent`** | Catalog verification, standard vs power user configuration validation, budget thresholds, approval level determination. | `tools_execution` |
| **`diagnostics_agent`**| Network/VPN triage, crash dump analysis, operating system specific commands, SLA target calculation. | `tools_execution` |
| **`tools_execution`** | Sandboxed dispatch of domain tools inside `try-except` error boundaries, converting outputs to structured records. | `responder_agent` or `failure_handler` |
| **`responder_agent`**  | Local LLM (Ollama) generation grounded strictly in retrieved context and tool findings, with deterministic fallback generator. | `memory_agent` or `failure_handler` |
| **`memory_agent`**     | Automatic vector embedding and indexing into ChromaDB episodic collection; stores user-taught facts into declarative store. | `END` |
| **`failure_handler`**  | Catches exceptions, network drops, or tool errors; generates safe guidance; logs incident trace; and recovers execution. | `memory_agent` |

---

## 3. Resilient Error Handling & Failure Recovery Taxonomy

To guarantee zero downtime and graceful degradation, the architecture implements multi-tier defensive boundaries:

```
[Level 1: Tool Isolation]
   └── Each tool in tools.py is wrapped in a try/except block.
       Returns ToolResult(success=False, error=str(e)) without crashing caller.

[Level 2: LangGraph Node Boundary]
   └── If any node raises an exception or populates state["error"],
       conditional edges immediately route execution to node_failure_handler.

[Level 3: LLM Disconnection Resilience]
   └── If Ollama is offline or times out (90s ceiling), responder_agent
       switches seamlessly to _generate_structured_fallback() using verified ChromaDB vectors.

[Level 4: Persistent DB Protection]
   └── ChromaDB operations employ stable hashing slugs for idempotency,
       preventing duplicate records or corrupt indexing upon repeated calls.

[Level 5: UI Error Boundary]
   └── Streamlit chat loops catch runtime anomalies, rendering a helpful
       remedial alert toast and emergency contact options rather than raw stack traces.
```

---

## 4. Domain Tools Library (`tools.py`)

1. **`search_policy_tool(rag, query, top_k)`**: Semantic search in `tech_policy_rag` collection with cosine similarity ranking.
2. **`ticket_lookup_tool(rag, query, ticket_id, limit)`**: Fast retrieval and filtering across `tech_tickets` collection.
3. **`ticket_creation_tool(rag, ticket_type, details)`**: Structured ticket generation in ChromaDB with live SLA calculation and priority classification.
4. **`hardware_catalog_tool(category, budget_max, keyword)`**: Direct specs, pricing, and approval rules for standard company hardware (MacBook Pro M3, ThinkPad X1, Dell Precision, monitors, docks, peripherals).
5. **`network_diagnostics_tool(issue_type, os_type)`**: Diagnostic matrices for GlobalProtect VPN, Corp-Secure 802.1X Wi-Fi, OS kernel panics/BSOD, and SSO app access.
6. **`sla_calculator_tool(urgency, issue_category)`**: Calculates first-response and target resolution timestamps based on priority levels.
7. **`memory_lookup_tool(rag, query, memory_type)`**: Semantic search across declarative rules and episodic interaction history.
8. **`system_status_tool()`**: Operational health checks of corporate IT services (Okta, GlobalProtect, Microsoft 365, Jira, AWS).

---

## 5. Persistent Vector Memory Architecture

```
Persistent Storage (.tech_agent_memory/)
├── tech_policy_rag
│     └── Embedded markdown sections from knowledge_base/tech_department_policies.md
│     └── Idempotent slug IDs: policy-{index}-{slug}
├── tech_tickets
│     └── Device request and issue tickets (TECH-YYYYMMDD-HEX)
│     └── Metadata: employee, department, urgency, priority, status, JSON payload
├── tech_declarative_memory
│     └── Permanent system operating rules and user-taught instructions
│     └── Uploaded document chunks (.pdf, .txt, .md) with page-level metadata
│     └── Vectorized with all-MiniLM-L6-v2 (384-d cosine embeddings)
└── tech_episodic_memory
      └── Interaction history auto-saved after every agent turn
      └── Vector content: "User asked: ... \nAgent answered: ..."
```

### Document & PDF Vector Ingestion Pipeline
- **Extractor**: Uses `pypdf.PdfReader` to extract clean text page-by-page.
- **Chunking Engine**: Segments documents into ~200-500 word semantic chunks with sliding overlap.
- **Enrichment**: Embeds chunk title, source filename, and page number for precise citations.
- **ChromaDB Storage**: Stored into `tech_declarative_memory` with stable doc IDs (`DOC-YYYYMMDD-HEX`).

---

## 6. Frontend & Voice Assistant Mode (`streamlit_app.py`)

- **Step Loading & Visual Progress**: Uses `st.status()` with live callbacks, rendering each agent's execution step in real-time.
- **ChatGPT-Style Unified Pill Bar**: Single consolidated bottom input bar (`st.chat_input`) supporting:
  - Text input with auto-expanding textarea
  - Audio dictation via native browser microphone (`SpeechRecognition` transcription)
  - Direct file attachment parsing (`.pdf`, `.txt`, `.md`, `.log`)
- **Voice Mode (TTS)**: Built-in Web Speech Synthesis button on assistant messages allowing users to listen to the agent's response aloud.
- **Tabbed Dashboard**:
  - **Support Desk**: Multi-agent chat, active agent badges, source citations, quick prompt chips.
  - **Ticket Central**: Interactive table view of all open/closed tickets with fast-track ticket creation directly inside the tab.
  - **Memory Vault**: Explorer for Policy vectors, Declarative rules, Uploaded Document chunks, and Episodic logs.
  - **Tools Sandbox**: Interactive testbed for all 8 domain tools.
  - **System Telemetry**: Real-time health indicators for Ollama, LangGraph, and ChromaDB collections.
