# ============================================================================
# agent_graph.py — LangGraph Multi-Agent Orchestration & Failure Handler
# ============================================================================
#
# This module implements the multi-agent state graph using LangGraph.
# It coordinates specialized agents through a structured, resilient workflow:
#
#   1. Triage Agent          — Analyzes user intent, urgency, and extracts entities
#   2. Policy Agent          — Specialist for corporate IT rules and compliance
#   3. Hardware Agent        — Specialist for device catalog, specs, and budget approvals
#   4. Diagnostics Agent     — Specialist for network, VPN, crash, and SLA calculation
#   5. Tools Execution Node  — Sandboxed execution of domain tools with error catching
#   6. Responder Agent       — Synthesizes tool findings with Ollama LLM / fallback
#   7. Memory Agent          — Persists conversation to episodic memory
#   8. Failure Handler Node  — Resilient error recovery node that handles any exception
#
# ============================================================================

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

import tools
from tools import ToolResult

logger = logging.getLogger("tech_agent.graph")


# ============================================================================
# Agent State Definition
# ============================================================================

class AgentState(TypedDict):
    """The shared state passed between all agents in the LangGraph workflow."""
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


# ============================================================================
# Multi-Agent Workflow Implementation
# ============================================================================

class TechMultiAgentWorkflow:
    """Orchestrates multi-agent routing, tool calling, memory logging, and failure handling."""

    def __init__(self, rag_instance: Any, llm_client: Any):
        """Initialize workflow with ChromaDB RAG and LLM client.
        
        Args:
            rag_instance: TechMemoryRAG instance.
            llm_client: OllamaClient instance.
        """
        self.rag = rag_instance
        self.llm = llm_client
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        """Construct the LangGraph StateGraph with nodes, edges, and error routing."""
        workflow = StateGraph(AgentState)

        # Register agent nodes
        workflow.add_node("triage_agent", self.node_triage_agent)
        workflow.add_node("policy_agent", self.node_policy_agent)
        workflow.add_node("hardware_agent", self.node_hardware_agent)
        workflow.add_node("diagnostics_agent", self.node_diagnostics_agent)
        workflow.add_node("tools_execution", self.node_tools_execution)
        workflow.add_node("responder_agent", self.node_responder_agent)
        workflow.add_node("memory_agent", self.node_memory_agent)
        workflow.add_node("failure_handler", self.node_failure_handler)

        # Set entry point
        workflow.set_entry_point("triage_agent")

        # Conditional routing after triage
        workflow.add_conditional_edges(
            "triage_agent",
            self._route_after_triage,
            {
                "route_policy": "policy_agent",
                "route_hardware": "hardware_agent",
                "route_diagnostics": "diagnostics_agent",
                "route_failure": "failure_handler",
            },
        )

        # Specialist agents route to tools execution
        workflow.add_edge("policy_agent", "tools_execution")
        workflow.add_edge("hardware_agent", "tools_execution")
        workflow.add_edge("diagnostics_agent", "tools_execution")

        # Conditional routing after tool execution (check for tool failure)
        workflow.add_conditional_edges(
            "tools_execution",
            self._route_after_tools,
            {
                "route_responder": "responder_agent",
                "route_failure": "failure_handler",
            },
        )

        # Responder routes to memory agent or failure handler if error
        workflow.add_conditional_edges(
            "responder_agent",
            self._route_after_responder,
            {
                "route_memory": "memory_agent",
                "route_failure": "failure_handler",
            },
        )

        # Failure handler routes to memory agent so the error/recovery is remembered
        workflow.add_edge("failure_handler", "memory_agent")

        # Memory agent finishes the workflow
        workflow.add_edge("memory_agent", END)

        return workflow.compile()

    # ========================================================================
    # Node 1: Triage & Routing Agent
    # ========================================================================
    def node_triage_agent(self, state: AgentState) -> AgentState:
        """Classify user intent, determine urgency level, and extract key entities.
        
        Intent classification:
            - "hardware_request": Laptop, monitor, purchase, upgrade, specs, mouse, keyboard
            - "device_issue": Crash, VPN, broken, freeze, BSOD, slow, bug, error, cannot connect
            - "policy_query": Policy, budget, allowance, rules, BYOD, security, approval, hours
        """
        trace = list(state.get("agent_trace", []))
        message = state.get("user_message", "").strip()
        msg_lower = message.lower()

        trace.append("🤖 [Triage Agent] Analyzing query intent, entities, and urgency...")

        try:
            # Urgency detection
            critical_keywords = ["lost", "stolen", "breach", "security", "production", "cannot work", "emergency", "fatal"]
            high_keywords = ["urgent", "vpn", "crash", "blocked", "meeting", "freeze", "down", "error"]

            if any(k in msg_lower for k in critical_keywords):
                urgency = "critical"
            elif any(k in msg_lower for k in high_keywords):
                urgency = "high"
            else:
                urgency = "normal"

            # Intent classification
            hardware_keywords = ["laptop", "macbook", "thinkpad", "dell", "monitor", "dock", "mouse", "keyboard", "spec", "headset", "request", "upgrade"]
            issue_keywords = ["vpn", "wifi", "connect", "crash", "error", "blue screen", "bsod", "broken", "frozen", "slow", "issue", "bug", "failing"]
            policy_keywords = ["policy", "rule", "budget", "allowance", "expense", "reimbursement", "wfh", "remote", "approval", "compliance"]

            if any(k in msg_lower for k in issue_keywords):
                intent = "device_issue"
            elif any(k in msg_lower for k in hardware_keywords):
                intent = "hardware_request"
            elif any(k in msg_lower for k in policy_keywords):
                intent = "policy_query"
            else:
                intent = "policy_query"  # Default to policy/knowledge retrieval

            # Entity extraction
            entities: Dict[str, Any] = {}
            if "macbook" in msg_lower or "apple" in msg_lower:
                entities["device_brand"] = "Apple"
            elif "thinkpad" in msg_lower or "lenovo" in msg_lower:
                entities["device_brand"] = "Lenovo"
            elif "dell" in msg_lower:
                entities["device_brand"] = "Dell"

            # Check for budget amounts (e.g. $2000, 2500 usd)
            budget_match = re.search(r"\$?\s*(\d{3,5})\s*(?:usd|dollars)?", msg_lower)
            if budget_match:
                entities["mentioned_budget"] = float(budget_match.group(1))

            # Detect explicit ticket creation requests
            ticket_action_patterns = [
                "raise ticket", "create ticket", "open ticket", "log ticket",
                "new ticket", "submit ticket", "report ticket", "file ticket",
                "raise a ticket", "log an incident", "report an issue",
                "file an incident", "make a ticket", "i want to raise a ticket",
                "create a ticket", "log a ticket"
            ]
            if any(p in msg_lower for p in ticket_action_patterns):
                entities["wants_ticket"] = True
                trace.append("🎫 [Triage Agent] Explicit ticket creation request detected.")

            trace.append(f"🎯 [Triage Agent] Intent detected: '{intent}' | Urgency: '{urgency.upper()}'")

            return {
                **state,
                "intent": intent,
                "urgency": urgency,
                "extracted_entities": entities,
                "active_agent": "Triage Agent",
                "agent_trace": trace,
                "error": None,
                "status": "in_progress",
            }
        except Exception as exc:
            logger.exception("Triage Agent failed: %s", exc)
            trace.append(f"⚠️ [Triage Agent] Encountered error: {str(exc)}")
            return {
                **state,
                "error": f"Triage Agent error: {str(exc)}",
                "agent_trace": trace,
                "active_agent": "Failure Handler",
            }

    # ========================================================================
    # Node 2: Policy & Compliance Specialist Agent
    # ========================================================================
    def node_policy_agent(self, state: AgentState) -> AgentState:
        """Formulate policy search queries and prepare tool requests."""
        trace = list(state.get("agent_trace", []))
        trace.append("📜 [Policy Agent] Consulting Corporate IT Knowledge Base & Policy vectors...")

        tool_calls = [
            {
                "tool": "search_policy_tool",
                "args": {"query": state["user_message"], "top_k": 4},
            }
        ]

        return {
            **state,
            "active_agent": "Policy Agent",
            "agent_trace": trace,
            "tool_calls": tool_calls,
        }

    # ========================================================================
    # Node 3: Hardware & Procurement Specialist Agent
    # ========================================================================
    def node_hardware_agent(self, state: AgentState) -> AgentState:
        """Formulate hardware catalog inquiries and policy checks."""
        trace = list(state.get("agent_trace", []))
        trace.append("💻 [Hardware Agent] Validating device specs, budget tiers, and procurement rules...")

        entities = state.get("extracted_entities", {})
        budget = entities.get("mentioned_budget")
        brand = entities.get("device_brand")

        tool_calls = [
            {
                "tool": "hardware_catalog_tool",
                "args": {"category": "all", "budget_max": budget, "keyword": brand},
            },
            {
                "tool": "search_policy_tool",
                "args": {"query": "device request equipment policy budget", "top_k": 2},
            },
        ]

        if entities.get("wants_ticket"):
            tool_calls.append({
                "tool": "ticket_creation_tool",
                "args": {
                    "ticket_type": "device_request",
                    "details": {
                        "employee": "Chat User",
                        "department": "General",
                        "device": brand or "Corporate Approved Device",
                        "business_reason": state["user_message"],
                    },
                },
            })
            trace.append("🎫 [Hardware Agent] Scheduled automated ticket registration into ChromaDB.")

        return {
            **state,
            "active_agent": "Hardware Agent",
            "agent_trace": trace,
            "tool_calls": tool_calls,
        }

    # ========================================================================
    # Node 4: IT Diagnostics & Support Specialist Agent
    # ========================================================================
    def node_diagnostics_agent(self, state: AgentState) -> AgentState:
        """Determine diagnostic playbook and calculate resolution SLA."""
        trace = list(state.get("agent_trace", []))
        trace.append("🛠️ [Diagnostics Agent] Running diagnostic matrix and computing SLA resolution targets...")

        msg = state["user_message"]
        urgency = state.get("urgency", "normal")
        entities = state.get("extracted_entities", {})

        tool_calls = [
            {
                "tool": "network_diagnostics_tool",
                "args": {"issue_type": msg, "os_type": "windows"},
            },
            {
                "tool": "sla_calculator_tool",
                "args": {"urgency": urgency, "issue_category": "network" if "vpn" in msg.lower() else "hardware"},
            },
            {
                "tool": "search_policy_tool",
                "args": {"query": f"device issue incident troubleshooting {msg}", "top_k": 2},
            },
        ]

        if entities.get("wants_ticket"):
            tool_calls.append({
                "tool": "ticket_creation_tool",
                "args": {
                    "ticket_type": "device_issue",
                    "details": {
                        "employee": "Chat User",
                        "device_asset": "Corporate Endpoint",
                        "urgency": urgency,
                        "issue_summary": msg,
                    },
                },
            })
            trace.append(f"🚨 [Diagnostics Agent] Scheduled automated incident registration ({urgency.upper()} SLA).")

        return {
            **state,
            "active_agent": "Diagnostics Agent",
            "agent_trace": trace,
            "tool_calls": tool_calls,
        }

    # ========================================================================
    # Node 5: Sandboxed Tools Execution Node
    # ========================================================================
    def node_tools_execution(self, state: AgentState) -> AgentState:
        """Execute all scheduled domain tools safely inside error boundaries."""
        trace = list(state.get("agent_trace", []))
        tool_calls = state.get("tool_calls", [])
        tool_results: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []

        trace.append(f"⚙️ [Tools Node] Executing {len(tool_calls)} domain tools...")

        for call in tool_calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})
            try:
                if tool_name == "search_policy_tool":
                    res = tools.search_policy_tool(self.rag, **args)
                    tool_results.append(res.to_dict())
                    if res.success and isinstance(res.output, list):
                        for item in res.output:
                            sources.append({
                                "id": item.get("id"),
                                "kind": "policy",
                                "similarity": item.get("similarity", 0.0),
                                "content": item.get("content", ""),
                                "title": item.get("title", ""),
                            })
                    trace.append(f"  ✓ search_policy_tool: found {len(res.output)} sections")

                elif tool_name == "hardware_catalog_tool":
                    res = tools.hardware_catalog_tool(**args)
                    tool_results.append(res.to_dict())
                    trace.append(f"  ✓ hardware_catalog_tool: returned {len(res.output)} devices")

                elif tool_name == "network_diagnostics_tool":
                    res = tools.network_diagnostics_tool(**args)
                    tool_results.append(res.to_dict())
                    trace.append(f"  ✓ network_diagnostics_tool: loaded playbook '{res.output.get('title')}'")

                elif tool_name == "sla_calculator_tool":
                    res = tools.sla_calculator_tool(**args)
                    tool_results.append(res.to_dict())
                    trace.append(f"  ✓ sla_calculator_tool: target resolution {res.output.get('full_resolution_target')}")

                elif tool_name == "ticket_lookup_tool":
                    res = tools.ticket_lookup_tool(self.rag, **args)
                    tool_results.append(res.to_dict())
                    trace.append(f"  ✓ ticket_lookup_tool: found {len(res.output)} tickets")

                elif tool_name == "ticket_creation_tool":
                    res = tools.ticket_creation_tool(self.rag, **args)
                    tool_results.append(res.to_dict())
                    if res.success:
                        trace.append(f"  ✓ ticket_creation_tool: created {res.output.get('ticket_id')} (SLA: {res.output.get('sla_target')})")
                    else:
                        trace.append(f"  ❌ ticket_creation_tool failed: {res.error}")

                else:
                    trace.append(f"  ⚠️ Unknown tool requested: {tool_name}")

            except Exception as exc:
                logger.exception("Error executing tool %s: %s", tool_name, exc)
                trace.append(f"  ❌ {tool_name} failed: {str(exc)}")
                tool_results.append({
                    "tool_name": tool_name,
                    "success": False,
                    "output": None,
                    "error": str(exc),
                })

        # Also search memory (declarative & episodic) to enrich context
        try:
            mem_res = self.rag.search(state["user_message"], top_k=2, include_memory=True)
            for m in mem_res:
                if m.kind in {"declarative", "episodic"}:
                    sources.append({
                        "id": m.id,
                        "kind": m.kind,
                        "similarity": m.similarity,
                        "content": m.content,
                        "title": m.kind.title(),
                    })
        except Exception as exc:
            logger.warning("Memory lookup in tool execution failed: %s", exc)

        return {
            **state,
            "tool_results": tool_results,
            "sources": sources,
            "agent_trace": trace,
            "active_agent": "Tools Executor",
        }

    # ========================================================================
    # Node 6: Synthesis & Responder Agent
    # ========================================================================
    def node_responder_agent(self, state: AgentState) -> AgentState:
        """Synthesize tool outputs and context into a polished user response."""
        trace = list(state.get("agent_trace", []))
        trace.append("💡 [Responder Agent] Synthesizing context and drafting response...")

        user_message = state["user_message"]
        tool_results = state.get("tool_results", [])
        sources = state.get("sources", [])

        # Check if Ollama is accessible
        ollama_ready = False
        try:
            ollama_ready = self.llm.is_available()
        except Exception as exc:
            logger.warning("Ollama availability check error: %s", exc)

        if ollama_ready:
            # Build structured prompt with tool results and policy context
            context_blocks = []
            for s in sources[:5]:
                context_blocks.append(f"[{s.get('kind', 'KNOWLEDGE').upper()} | {s.get('title', '')}]\n{s.get('content', '')}")

            for tr in tool_results:
                if tr.get("success") and tr.get("output"):
                    context_blocks.append(f"[TOOL: {tr.get('tool_name')}]\n{tr.get('output')}")

            context_str = "\n\n".join(context_blocks)
            prompt = f"""You are the senior Corporate Tech Support Agent.
Answer the employee professionally, accurately, and concisely using the provided context and tool outputs.
If details are missing (e.g. employee name, manager approval, asset tag), clearly ask for them.
Adhere strictly to company policy without hallucinating.

Context & Tool Findings:
{context_str}

Employee Inquiry:
{user_message}

Answer:"""
            try:
                answer = self.llm.generate(prompt)
                trace.append("  ✓ Generated answer via local LLM (Ollama)")
            except Exception as exc:
                logger.warning("Ollama generation failed: %s. Using structured fallback.", exc)
                trace.append("  ⚠️ LLM request timed out or dropped; falling back to structured synthesis.")
                answer = self._generate_structured_fallback(state)
        else:
            trace.append("  ℹ️ Local LLM (Ollama) offline; delivering grounded response from verified tools & policy vectors.")
            answer = self._generate_structured_fallback(state)

        return {
            **state,
            "final_answer": answer,
            "agent_trace": trace,
            "active_agent": "Responder Agent",
            "status": "completed",
        }

    # ========================================================================
    # Node 7: Memory & Knowledge Agent
    # ========================================================================
    def node_memory_agent(self, state: AgentState) -> AgentState:
        """Persist interaction to episodic memory and update declarative facts."""
        trace = list(state.get("agent_trace", []))
        trace.append("💾 [Memory Agent] Persisting interaction to ChromaDB episodic vector collection...")

        query = state.get("user_message", "")
        answer = state.get("final_answer", "")
        intent = state.get("intent", "general")

        try:
            mem_id = self.rag.remember_interaction(
                query=query,
                answer=answer,
                category=intent,
            )
            trace.append(f"  ✓ Saved to episodic memory as {mem_id}")
        except Exception as exc:
            logger.exception("Failed to save episodic memory: %s", exc)
            trace.append(f"  ⚠️ Warning: Memory saving failed ({str(exc)})")

        return {
            **state,
            "agent_trace": trace,
            "active_agent": "Memory Agent",
            "status": "completed" if not state.get("error") else "recovered",
        }

    # ========================================================================
    # Node 8: Resilient Failure & Error Recovery Handler
    # ========================================================================
    def node_failure_handler(self, state: AgentState) -> AgentState:
        """Catch any system or node failures, formulate recovery response, and prevent crashes."""
        trace = list(state.get("agent_trace", []))
        error_msg = state.get("error", "Unknown system exception")

        trace.append(f"🛡️ [Failure Handler] Intercepted execution fault: {error_msg}")
        trace.append("🛡️ [Failure Handler] Initiating automatic recovery protocol...")

        # Formulate self-healing response for the user
        recovery_response = (
            f"### 🛡️ Support Agent Recovery Mode\n\n"
            f"An unexpected issue occurred while processing your request: `{error_msg}`.\n\n"
            f"**Safe Fallback Information:**\n"
            f"- For immediate hardware requests, please submit the form in the sidebar with your Manager's approval.\n"
            f"- For urgent IT outages or VPN blocks, contact the **24/7 IT Emergency Desk** at `helpdesk@company.internal` or call extension **#4357 (HELP)**.\n"
            f"- Our persistent memory and ChromaDB collections remain safe and intact."
        )

        return {
            **state,
            "final_answer": recovery_response,
            "agent_trace": trace,
            "active_agent": "Failure Handler",
            "status": "recovered",
            "error": None,  # Cleared after handling
        }

    # ========================================================================
    # Routing Conditions
    # ========================================================================
    def _route_after_triage(self, state: AgentState) -> str:
        if state.get("error"):
            return "route_failure"
        intent = state.get("intent", "policy_query")
        if intent == "hardware_request":
            return "route_hardware"
        elif intent == "device_issue":
            return "route_diagnostics"
        return "route_policy"

    def _route_after_tools(self, state: AgentState) -> str:
        if state.get("error"):
            return "route_failure"
        return "route_responder"

    def _route_after_responder(self, state: AgentState) -> str:
        if state.get("error"):
            return "route_failure"
        return "route_memory"

    # ========================================================================
    # Structured Fallback Generator
    # ========================================================================
    def _generate_structured_fallback(self, state: AgentState) -> str:
        """Create a well-formatted markdown response from tool findings when LLM is unavailable."""
        intent = state.get("intent", "policy_query")
        tool_results = state.get("tool_results", [])
        sources = state.get("sources", [])

        lines = []

        # If a ticket was created by domain tools, headline it prominently
        t_res = next((r for r in tool_results if r.get("tool_name") == "ticket_creation_tool"), None)
        if t_res and t_res.get("success") and t_res.get("output"):
            t_data = t_res["output"]
            lines.append(f"### 🎫 Corporate IT Ticket Registered: `{t_data.get('ticket_id')}`\n")
            lines.append(f"- **Ticket Type:** `{t_data.get('ticket_type', '').upper()}` | **Status:** `OPEN (ChromaDB Indexed)`")
            lines.append(f"- **Target Resolution Window:** {t_data.get('sla_target')}")
            lines.append(f"- **Initial Response Window:** {t_data.get('first_response')}")
            lines.append(f"- **Assigned Engineering Tier:** {t_data.get('assigned_tier')}")
            lines.append("\n*Your ticket has been officially logged and prioritized. Track updates in Ticket Central.*\n\n---")

        if intent == "hardware_request":
            lines.append("### 💻 Hardware Catalog & Procurement Guidance\n")
            hw_res = next((r for r in tool_results if r.get("tool_name") == "hardware_catalog_tool"), None)
            if hw_res and hw_res.get("success") and hw_res.get("output"):
                lines.append("**Approved Standard Models Matching Your Criteria:**")
                for item in hw_res["output"][:3]:
                    lines.append(f"- **{item['name']}** ({item['cost']})")
                    lines.append(f"  - *Specs:* {item['specs']}")
                    lines.append(f"  - *Lead Time:* {item['lead_time']} | *Rule:* {item['approval_rule']}")
            lines.append("\n**Next Steps:** Use the **Device Request** form in the sidebar with your business reason and manager's name.")

        elif intent == "device_issue":
            lines.append("### 🛠️ IT Diagnostics & Troubleshooting Plan\n")
            diag_res = next((r for r in tool_results if r.get("tool_name") == "network_diagnostics_tool"), None)
            if diag_res and diag_res.get("success") and diag_res.get("output"):
                diag = diag_res["output"]
                lines.append(f"**Playbook:** {diag.get('title')}")
                lines.append("**Immediate Recommended Steps:**")
                for s in diag.get("recommended_steps", [])[:4]:
                    lines.append(f"- {s}")
                lines.append(f"\n*Escalation:* {diag.get('escalation_team')}")

            sla_res = next((r for r in tool_results if r.get("tool_name") == "sla_calculator_tool"), None)
            if sla_res and sla_res.get("success") and sla_res.get("output"):
                sla = sla_res["output"]
                lines.append(f"\n**SLA Commitment:** {sla.get('full_resolution_target')} ({sla.get('assigned_tier')})")

        else:
            lines.append("### 📜 Company IT Policy Findings\n")
            if sources:
                lines.append("**Relevant Policies:**")
                for s in sources[:3]:
                    content_snippet = s.get("content", "").split("\n")[0]
                    lines.append(f"- **{s.get('title') or 'Policy'}**: {content_snippet}")
            else:
                lines.append("No exact policy match found in the corporate database. Please verify your query.")

            lines.append("\n*Need immediate escalation? Type `/help` or use the sidebar forms.*")

        return "\n".join(lines)

    # ========================================================================
    # Public Execution Method
    # ========================================================================
    def run(self, user_message: str, step_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """Execute the LangGraph multi-agent workflow for a user message.
        
        Args:
            user_message: The query submitted by the user.
            step_callback: Optional callback invoked with each step message for live UI loading.
            
        Returns:
            Dictionary containing final answer, sources, agent trace, and tool results.
        """
        initial_state: AgentState = {
            "user_message": user_message,
            "intent": "policy_query",
            "urgency": "normal",
            "extracted_entities": {},
            "active_agent": "Triage Agent",
            "agent_trace": [],
            "tool_calls": [],
            "tool_results": [],
            "sources": [],
            "final_answer": "",
            "error": None,
            "status": "in_progress",
        }

        try:
            # Stream execution if step_callback is provided
            final_state = initial_state
            for chunk in self.graph.stream(initial_state):
                for node_name, node_state in chunk.items():
                    final_state = node_state
                    if step_callback and node_state.get("agent_trace"):
                        latest_step = node_state["agent_trace"][-1]
                        step_callback(latest_step)

            return {
                "answer": final_state.get("final_answer", ""),
                "sources": final_state.get("sources", []),
                "agent_trace": final_state.get("agent_trace", []),
                "active_agent": final_state.get("active_agent", "Agent"),
                "tool_results": final_state.get("tool_results", []),
                "intent": final_state.get("intent", ""),
                "urgency": final_state.get("urgency", ""),
                "status": final_state.get("status", "completed"),
            }
        except Exception as exc:
            logger.exception("LangGraph workflow error: %s", exc)
            # Run failure handler directly if graph stream threw an unhandled exception
            failed_state = {**initial_state, "error": str(exc)}
            recovered = self.node_failure_handler(failed_state)
            return {
                "answer": recovered["final_answer"],
                "sources": [],
                "agent_trace": recovered["agent_trace"],
                "active_agent": "Failure Handler",
                "tool_results": [],
                "intent": "error_recovery",
                "urgency": "high",
                "status": "recovered",
            }
