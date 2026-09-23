# ============================================================================
# streamlit_app.py — Modern UI/UX for Tech Department Multi-Agent Support
# ============================================================================
#
# This module provides a high-fidelity web dashboard featuring:
#   1. LangGraph Multi-Agent Orchestration with live step loading & visual progress
#   2. Resilient Error & Failure Handler recovery feedback
#   3. Integrated Domain Tools (Hardware Catalog, Diagnostics, SLA Calculator)
#   4. Voice Mode: Speech-to-Text input + Text-to-Speech audio read-aloud
#   5. Tabbed Workspace:
#      - 💬 Support Desk (Interactive multi-agent chat + live reasoning trace)
#      - 🎫 Ticket Central (Search, inspect, and filter company tickets)
#      - 🧠 Memory & Knowledge Vault (Inspect policy vectors, episodic & declarative)
#      - 🛠️ IT Diagnostic Tools Sandbox (Run troubleshooting tools directly)
#      - 📊 System Telemetry & Health (ChromaDB, Ollama, and Agent latency)
#
# Run with: streamlit run streamlit_app.py --server.fileWatcherType none
# ============================================================================

from __future__ import annotations

import html
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Suppress noisy library logs
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import streamlit as st
import tools
from tech_agent import OllamaClient, TechDepartmentAgent, TechMemoryRAG

logger = logging.getLogger("tech_agent.ui")

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="TechCore | AI IT Support Hub",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Premium Corporate CSS Styling
# ============================================================================
st.markdown(
    """
    <style>
        /* Global Font & Theme */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Hero Banner */
        .hero-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
        }

        .hero-title {
            font-size: 1.75rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .hero-subtitle {
            color: #94a3b8;
            font-size: 0.92rem;
            margin-top: 6px;
            margin-bottom: 0;
        }

        /* Glowing Status Dot */
        .status-dot {
            height: 10px;
            width: 10px;
            background-color: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 10px #10b981;
            margin-right: 6px;
        }

        /* Agent Badge */
        .agent-pill {
            display: inline-flex;
            align-items: center;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #38bdf8;
            margin-bottom: 8px;
        }

        .agent-pill.recovered {
            background: rgba(245, 158, 11, 0.15);
            border-color: rgba(245, 158, 11, 0.4);
            color: #fbbf24;
        }

        /* Glassmorphism Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 10px;
            padding: 14px 16px;
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(56, 189, 248, 0.4);
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .metric-label {
            font-size: 0.76rem;
            font-weight: 500;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Citation / Source Cards */
        .source-box {
            background: rgba(15, 23, 42, 0.6);
            border-left: 3px solid #38bdf8;
            border-radius: 6px;
            padding: 10px 14px;
            margin-top: 8px;
            font-size: 0.88rem;
        }

        /* Speech Synthesis Button */
        .tts-button {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.3);
            color: #38bdf8;
            border-radius: 6px;
            padding: 4px 10px;
        /* High-Contrast Corporate Theme & Sidebar Polish */
        [data-testid="stSidebar"] {
            background-color: #0b1120 !important;
            border-right: 1px solid rgba(56, 189, 248, 0.2) !important;
        }

        .sidebar-header-title {
            font-weight: 800;
            font-size: 1.18rem;
            color: #38bdf8 !important;
            letter-spacing: -0.01em;
        }

        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3 {
            color: #f8fafc !important;
            font-weight: 700;
        }

        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] label {
            color: #cbd5e1 !important;
        }

        [data-testid="stSidebar"] .metric-card {
            background: rgba(15, 23, 42, 0.95) !important;
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
            border-radius: 10px;
            padding: 12px 14px;
            text-align: center;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }

        [data-testid="stSidebar"] .metric-value {
            font-size: 1.6rem !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
        }

        [data-testid="stSidebar"] .metric-label {
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: #94a3b8 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* ====================================================================
           ChatGPT-Style Floating Pill Chat Input Bar
           ==================================================================== */
        [data-testid="stBottom"] {
            background: linear-gradient(180deg, rgba(11, 15, 25, 0) 0%, rgba(11, 15, 25, 0.95) 40%, rgba(11, 15, 25, 1) 100%) !important;
            padding-bottom: 20px !important;
        }

        [data-testid="stBottom"] > div {
            max-width: 840px !important;
            margin: 0 auto !important;
        }

        [data-testid="stChatInput"] {
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 9999px !important;
            background: #212121 !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6) !important;
            padding: 4px 14px !important;
            transition: all 0.2s ease !important;
            align-items: center !important;
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: rgba(56, 189, 248, 0.6) !important;
            box-shadow: 0 8px 32px rgba(56, 189, 248, 0.18) !important;
        }

        [data-testid="stChatInputTextArea"] {
            background: transparent !important;
            color: #f3f4f6 !important;
            font-size: 15px !important;
            padding: 8px 12px !important;
        }

        [data-testid="stChatInputTextArea"]::placeholder {
            color: #9ca3af !important;
            font-size: 15px !important;
        }

        /* Round Blue Circle Send Button (Identical to ChatGPT) */
        [data-testid="stChatInputSubmitButton"] {
            background: #2563eb !important;
            border-radius: 50% !important;
            width: 36px !important;
            height: 36px !important;
            min-width: 36px !important;
            min-height: 36px !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            border: none !important;
            box-shadow: 0 2px 10px rgba(37, 99, 235, 0.5) !important;
            transition: transform 0.15s ease, background 0.15s ease !important;
        }

        [data-testid="stChatInputSubmitButton"]:hover {
            background: #1d4ed8 !important;
            transform: scale(1.08) !important;
        }

        [data-testid="stChatInputSubmitButton"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
            width: 18px !important;
            height: 18px !important;
        }

        /* Microphone Button inside the Pill */
        [data-testid="stChatInput"] button[aria-label*="audio" i],
        [data-testid="stChatInput"] button[aria-label*="record" i],
        [data-testid="stChatInput"] button:not([data-testid="stChatInputSubmitButton"]):not([data-testid="stChatInputFileUploadButton"]) {
            color: #cbd5e1 !important;
            border-radius: 50% !important;
            width: 34px !important;
            height: 34px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: color 0.15s ease, background 0.15s ease !important;
        }

        [data-testid="stChatInput"] button[aria-label*="audio" i]:hover,
        [data-testid="stChatInput"] button[aria-label*="record" i]:hover {
            color: #38bdf8 !important;
            background: rgba(56, 189, 248, 0.15) !important;
        }

        /* Plus Attachment Button on Left */
        [data-testid="stChatInputFileUploadButton"],
        [data-testid="stChatInput"] button[aria-label*="attach" i],
        [data-testid="stChatInput"] button[aria-label*="upload" i] {
            color: #94a3b8 !important;
            border-radius: 50% !important;
            transition: color 0.15s ease !important;
        }

        [data-testid="stChatInputFileUploadButton"]:hover {
            color: #f8fafc !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Cached Agent Resource Loader
# ============================================================================
@st.cache_resource(show_spinner="Initializing TechCore Multi-Agent Orchestrator...")
def load_support_system(model_name: str, base_url: str, version: str = "v4.0-pdf-kb") -> TechDepartmentAgent:
    """Initialize ChromaDB RAG, Ollama Client, and LangGraph Multi-Agent system."""
    try:
        rag = TechMemoryRAG()
        llm = OllamaClient(model_name=model_name, base_url=base_url)
        # Automatically verify and start Ollama server in background if not running
        llm.ensure_running()
        return TechDepartmentAgent(rag=rag, llm=llm)
    except Exception as exc:
        logger.exception("Fatal error loading support system: %s", exc)
        st.error(f"Critical error initializing support system: {str(exc)}")
        raise


# ============================================================================
# Audio Helper Functions (Speech-to-Text & Text-to-Speech)
# ============================================================================
def render_tts_player(text: str, message_id: str) -> None:
    """Render a Web Speech API Text-to-Speech audio button for reading answers aloud."""
    clean_text = text.replace('"', '\\"').replace("\n", " ").replace("\r", " ")
    tts_html = f"""
    <div style="margin-top: 6px;">
        <button id="btn_{message_id}" onclick="playTTS_{message_id}()" 
                style="background: rgba(56, 189, 248, 0.12); border: 1px solid rgba(56, 189, 248, 0.35); color: #38bdf8; padding: 4px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px;">
            🔊 <span>Listen to Answer</span>
        </button>
        <button id="btn_stop_{message_id}" onclick="stopTTS()" 
                style="background: rgba(244, 63, 94, 0.12); border: 1px solid rgba(244, 63, 94, 0.35); color: #f43f5e; padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; display: none;">
            ⏹️ Stop
        </button>
    </div>
    <script>
        function playTTS_{message_id}() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
                var utterance = new SpeechSynthesisUtterance("{clean_text[:600]}");
                utterance.rate = 1.0;
                utterance.pitch = 1.0;
                var stopBtn = document.getElementById("btn_stop_{message_id}");
                stopBtn.style.display = "inline-flex";
                utterance.onend = function() {{
                    stopBtn.style.display = "none";
                }};
                utterance.onerror = function() {{
                    stopBtn.style.display = "none";
                }};
                window.speechSynthesis.speak(utterance);
            }} else {{
                alert("Text-to-speech is not supported by your current browser.");
            }}
        }}
        function stopTTS() {{
            if ('speechSynthesis' in window) {{
                window.speechSynthesis.cancel();
            }}
        }}
    </script>
    """
    if hasattr(st, "iframe"):
        st.iframe(tts_html, height=45)
    else:
        st.components.v1.html(tts_html, height=45)


def transcribe_audio_bytes(audio_bytes: bytes) -> Optional[str]:
    """Transcribe microphone audio recorded via native st.chat_input into text."""
    try:
        import io
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text.strip()
    except Exception as exc:
        logger.warning("Native speech recognition failed: %s", exc)
        return None



# Ensure messages session state is initialized before any sidebar form submissions
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Hello! I am your TechCore Multi-Agent IT Assistant.**\n\n"
                "I am powered by a LangGraph multi-agent team (Triage, Policy, Hardware, Diagnostics, "
                "Responder, Memory, and Failure Recovery).\n\n"
                "Ask me anything about **company IT policies, hardware requests, VPN/network issues, "
                "or software licenses** — or click 🎙️ **Voice Mode** right at the input bar to speak!"
            ),
            "agent_trace": ["🤖 [Triage Agent] Initialized multi-agent orchestrator"],
            "active_agent": "TechCore Orchestrator",
            "sources": [],
            "status": "completed",
        }
    ]

# ============================================================================
# SIDEBAR — Configuration, Metrics, and Forms
# ============================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
            <div class="status-dot"></div>
            <span class="sidebar-header-title">TechCore Agent Engine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # LLM Settings
    with st.expander("⚙️ LLM & Engine Config", expanded=False):
        model_name = st.text_input("Ollama Model", value="mistral")
        base_url = st.text_input("Ollama Endpoint", value="http://localhost:11434")

    # Load agent safely with cache bust safeguard
    try:
        agent = load_support_system(model_name, base_url, version="v4.0-pdf-kb")
        if not hasattr(agent, "ingest_document_bytes"):
            st.cache_resource.clear()
            agent = load_support_system(model_name, base_url, version="v4.1-pdf-kb")
        stats = agent.rag.stats()
    except Exception as exc:
        st.error(f"Failed to load agent: {exc}")
        st.stop()

    # Telemetry metrics
    st.markdown("### 📊 Vector Memory Telemetry")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats.get('policies', 0)}</div>
                <div class="metric-label">Policy Vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 10px;">
                <div class="metric-value">{stats.get('episodic_memories', 0)}</div>
                <div class="metric-label">Episodic Logs</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{stats.get('tickets', 0)}</div>
                <div class="metric-label">Total Tickets</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 10px;">
                <div class="metric-value">{stats.get('declarative_memories', 0)}</div>
                <div class="metric-label">Saved Facts</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Form 1: Teach Declarative Knowledge & Upload Documents
    with st.expander("🧠 Teach Agent Knowledge", expanded=False):
        st.caption("Add business rules or upload corporate policy documents (PDF, TXT, MD).")
        with st.form("teach_fact_form", clear_on_submit=True):
            fact_text = st.text_area(
                "Fact or Operating Rule",
                placeholder="e.g. Always route executive hardware upgrades to VP IT directly.",
            )
            uploaded_doc = st.file_uploader(
                "📄 Upload Knowledge / Policy PDF",
                type=["pdf", "txt", "md"],
                help="Upload company SOPs, manuals, or PDF policies to vectorize into ChromaDB.",
            )
            submit_fact = st.form_submit_button("💾 Save Knowledge & Index Document")

        if submit_fact:
            if uploaded_doc is not None:
                with st.spinner(f"Memory Agent parsing and vectorizing {uploaded_doc.name}..."):
                    ingest_fn = getattr(agent, "ingest_document_bytes", None) or getattr(agent.rag, "ingest_document_bytes", None)
                    if not ingest_fn:
                        st.cache_resource.clear()
                        agent = load_support_system(model_name, base_url, version="v4.2-pdf-reload")
                        ingest_fn = getattr(agent, "ingest_document_bytes", None) or getattr(agent.rag, "ingest_document_bytes", None)
                    
                    res = ingest_fn(
                        file_bytes=uploaded_doc.getvalue(),
                        filename=uploaded_doc.name,
                    )
                    if res.get("success"):
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": f"📄 **Uploaded & Ingested Document:** `{uploaded_doc.name}`" + (f"\n> Note: \"{fact_text.strip()}\"" if fact_text.strip() else ""),
                            }
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": res["answer"],
                                "agent_trace": res.get("agent_trace", []),
                                "active_agent": res.get("active_agent", "Memory Agent"),
                                "sources": [],
                                "status": "completed",
                            }
                        )
                        st.toast(f"📄 Stored in knowledge_base/ & vectorized: {uploaded_doc.name}", icon="✅")
                        st.rerun()
                    else:
                        st.error(res.get("answer", "Failed to process document."))

            elif fact_text.strip():
                with st.spinner("Memory Agent embedding and indexing in declarative memory..."):
                    res = agent.teach_fact(fact_text.strip())
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": f"🧠 **Teach Policy Rule / Fact to Agent:**\n> \"{fact_text.strip()}\"",
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res["answer"],
                            "agent_trace": res.get("agent_trace", []),
                            "active_agent": res.get("active_agent", "Memory Agent"),
                            "sources": [],
                            "status": "completed",
                        }
                    )
                    st.toast(f"🧠 Knowledge rule saved: {res['memory_id']}", icon="💡")
                    st.rerun()
            else:
                st.warning("Please provide a fact or select a PDF / document to upload.")

    # Form 2: Submit Device Request
    with st.expander("💻 Submit Device Request", expanded=False):
        with st.form("sidebar_device_request", clear_on_submit=True):
            req_emp = st.text_input("Employee Name *")
            req_dept = st.text_input("Department *")
            req_mgr = st.text_input("Approving Manager *")
            req_dev = st.selectbox(
                "Hardware Selection",
                [
                    'Apple MacBook Pro 14" (M3 Pro, 36GB)',
                    'Apple MacBook Air 15" (M3, 16GB)',
                    'Lenovo ThinkPad X1 Carbon Gen 12 (32GB)',
                    'Dell Precision 5690 Workstation (64GB)',
                    'Dell UltraSharp 27" 4K Monitor',
                    'CalDigit TS4 Thunderbolt Dock',
                    'Logitech MX Master 3S Combo',
                    'Jabra Evolve2 75 ANC Headset',
                ],
            )
            req_mode = st.selectbox("Work Location Mode", ["Hybrid", "Remote", "Office"])
            req_reason = st.text_area("Business Justification *")
            req_date = st.date_input("Target Date")
            submit_dev = st.form_submit_button("🚀 Submit Device Request")

        if submit_dev:
            if req_emp and req_dept and req_mgr and req_reason:
                with st.spinner("Hardware Specialist evaluating procurement & SLAs..."):
                    res = agent.submit_device_request(
                        {
                            "employee": req_emp,
                            "department": req_dept,
                            "manager": req_mgr,
                            "device": req_dev,
                            "work_mode": req_mode,
                            "business_reason": req_reason,
                            "preferred_date": str(req_date),
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"📋 **Submitted Hardware Request:**\n"
                                f"- **Device:** {req_dev}\n"
                                f"- **Employee:** {req_emp} ({req_dept})\n"
                                f"- **Approving Manager:** {req_mgr}\n"
                                f"- **Work Mode:** {req_mode}\n"
                                f"- **Target Date:** {req_date}\n"
                                f"- **Business Justification:** \"{req_reason}\""
                            ),
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res["answer"],
                            "agent_trace": res.get("agent_trace", []),
                            "active_agent": res.get("active_agent", "Hardware Specialist"),
                            "sources": [],
                            "status": "completed",
                        }
                    )
                    st.toast(f"🎫 Hardware ticket dispatched: {res['ticket_id']}", icon="✅")
                    st.rerun()
            else:
                st.error("Please fill in all required fields marked with *.")

    # Form 3: Report Device Issue
    with st.expander("🛠️ Report Technical Issue", expanded=False):
        with st.form("sidebar_device_issue", clear_on_submit=True):
            iss_emp = st.text_input("Employee Name *")
            iss_tag = st.text_input("Device / Asset Tag *")
            iss_urgency = st.selectbox("Urgency", ["Normal", "High", "Critical"])
            iss_summary = st.text_area("Issue Description *", placeholder="What went wrong?")
            iss_steps = st.text_area("Troubleshooting Attempted", placeholder="Restarted, cleared cache, etc.")
            submit_iss = st.form_submit_button("⚠️ Log Incident Ticket")

        if submit_iss:
            if iss_emp and iss_tag and iss_summary:
                with st.spinner("Diagnostics Specialist auto-triaging priority & SLAs..."):
                    res = agent.submit_device_issue(
                        {
                            "employee": iss_emp,
                            "device_asset": iss_tag,
                            "urgency": iss_urgency,
                            "issue_summary": iss_summary,
                            "troubleshooting": iss_steps,
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"⚠️ **Reported Technical Incident:**\n"
                                f"- **Affected Asset:** `{iss_tag}`\n"
                                f"- **Employee:** {iss_emp}\n"
                                f"- **Classified Urgency:** `{iss_urgency.upper()}`\n"
                                f"- **Issue Summary:** \"{iss_summary}\"\n"
                                f"- **Troubleshooting:** \"{iss_steps or 'None'}\""
                            ),
                        }
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": res["answer"],
                            "agent_trace": res.get("agent_trace", []),
                            "active_agent": res.get("active_agent", "Diagnostics Specialist"),
                            "sources": [],
                            "status": "completed",
                        }
                    )
                    st.toast(f"🚨 Incident logged: {res['ticket_id']}", icon="🛠️")
                    st.rerun()
            else:
                st.error("Please fill in all required fields marked with *.")

    # Reset memory button
    st.divider()
    if st.button("🗑️ Clear Episodic & Ticket Memory", type="secondary"):
        agent.rag.clear_memory()
        st.toast("Memory reset to default state.", icon="🔄")
        st.rerun()


# ============================================================================
# MAIN INTERFACE — Hero Header & Tabbed Workspace
# ============================================================================
st.markdown(
    """
    <div class="hero-header">
        <h1 class="hero-title">🛡️ TechCore Support Hub</h1>
        <p class="hero-subtitle">
            LangGraph Multi-Agent Orchestration • 4 ChromaDB Vector Stores • Self-Healing Failure Handling • Voice Mode
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_chat, tab_tickets, tab_memory, tab_tools, tab_system = st.tabs(
    [
        "💬 Support Desk",
        "🎫 Ticket Central",
        "🧠 Memory & Knowledge Vault",
        "🛠️ Diagnostic Tools Sandbox",
        "📊 System Telemetry & Health",
    ]
)


# ============================================================================
# TAB 1: Support Desk (Interactive Chat with Multi-Agent Reasoning Trace)
# ============================================================================
with tab_chat:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Hello! I am your TechCore Multi-Agent IT Assistant.**\n\n"
                    "I am powered by a LangGraph multi-agent team (Triage, Policy, Hardware, Diagnostics, "
                    "Responder, Memory, and Failure Recovery).\n\n"
                    "Ask me anything about **company IT policies, hardware requests, VPN/network issues, "
                    "or software licenses** — or click 🎙️ **Voice Mode** right at the input bar to speak!"
                ),
                "agent_trace": ["🤖 [Triage Agent] Initialized multi-agent orchestrator"],
                "active_agent": "TechCore Orchestrator",
                "sources": [],
                "status": "completed",
            }
        ]

    # Quick prompt chips for fast testing
    st.markdown("**Quick Inquiries:**")
    col_q1, col_q2, col_q3, col_q4 = st.columns(4)
    quick_query = None
    with col_q1:
        if st.button("💻 Laptop Budget Policy", use_container_width=True):
            quick_query = "What is the company budget and policy for requesting a new laptop?"
    with col_q2:
        if st.button("🔒 VPN Drop Diagnostics", use_container_width=True):
            quick_query = "My GlobalProtect corporate VPN is disconnecting with error 403. How to fix?"
    with col_q3:
        if st.button("⚙️ Dell Workstation Specs", use_container_width=True):
            quick_query = "What are the specs, cost, and approval requirements for Dell Precision Workstations?"
    with col_q4:
        if st.button("🚨 Report Stolen Laptop", use_container_width=True):
            quick_query = "My company MacBook was lost during travel. What critical steps must I take?"

    # Check for voice input via query params (from ChatGPT Voice Bar)
    voice_query = st.query_params.get("voice_input")
    if voice_query:
        st.query_params.clear()
        st.session_state.pending_prompt = voice_query
    elif quick_query:
        st.session_state.pending_prompt = quick_query

    # Render previous messages
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            # If assistant message has an active agent, show badge
            if msg["role"] == "assistant" and msg.get("active_agent"):
                badge_class = "recovered" if msg.get("status") == "recovered" else ""
                st.markdown(
                    f"""<div class="agent-pill {badge_class}">Active Agent: {msg.get('active_agent')}</div>""",
                    unsafe_allow_html=True,
                )

            st.markdown(msg["content"])

            # Render Voice TTS player for assistant messages
            if msg["role"] == "assistant" and len(msg["content"]) > 15:
                render_tts_player(msg["content"], f"msg_{idx}")

            # Expandable agent reasoning trace
            trace = msg.get("agent_trace", [])
            if trace and len(trace) > 1:
                with st.expander("🔍 View Multi-Agent Execution Trace", expanded=False):
                    for step in trace:
                        st.code(step, language="text")

            # Expandable retrieved knowledge sources
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📚 Retrieved Context ({len(sources)} sources)", expanded=False):
                    for s in sources[:4]:
                        s_kind = getattr(s, "kind", "policy")
                        s_sim = getattr(s, "similarity", 0.85)
                        s_id = getattr(s, "id", "doc")
                        s_content = getattr(s, "content", "")
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <strong>[{s_kind.upper()}]</strong> <code>{s_id}</code> | Relevance: <code>{s_sim:.2f}</code>
                                <p style="margin-top: 4px; color: #cbd5e1;">{html.escape(s_content[:300])}...</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Process pending prompt (from typing, quick buttons, or voice mode)
    if st.session_state.get("pending_prompt"):
        active_prompt = st.session_state.pop("pending_prompt")
        
        # Display user message
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Assistant generation with dynamic step loading
        with st.chat_message("assistant"):
            step_container = st.status(
                "🤖 LangGraph Multi-Agent Team Dispatching...", expanded=True
            )

            steps_captured: List[str] = []

            def on_agent_step(step_text: str):
                steps_captured.append(step_text)
                step_container.write(step_text)

            try:
                # Execute multi-agent graph with step callback
                try:
                    result = agent.answer(active_prompt, step_callback=on_agent_step)
                except TypeError:
                    result = agent.answer(active_prompt)

                final_status = result.get("status", "completed")
                active_agent = result.get("active_agent", "Tech Agent")

                if final_status == "recovered":
                    step_container.update(
                        label=f"🛡️ Recovered gracefully via {active_agent}",
                        state="complete",
                        expanded=False,
                    )
                else:
                    step_container.update(
                        label=f"✅ {active_agent} synthesized response",
                        state="complete",
                        expanded=False,
                    )

            except Exception as exc:
                logger.exception("Unexpected error in chat answering: %s", exc)
                step_container.update(
                    label=f"⚠️ Handled error: {str(exc)}", state="error", expanded=False
                )
                result = {
                    "answer": f"I encountered a temporary problem: `{str(exc)}`. Please refer to company helpdesk at `helpdesk@company.internal`.",
                    "sources": [],
                    "agent_trace": steps_captured + [f"❌ Error: {str(exc)}"],
                    "active_agent": "Failure Handler",
                    "status": "recovered",
                }

            # Render response
            badge_class = "recovered" if result.get("status") == "recovered" else ""
            st.markdown(
                f"""<div class="agent-pill {badge_class}">Active Agent: {result.get('active_agent', 'Tech Agent')}</div>""",
                unsafe_allow_html=True,
            )
            st.markdown(result["answer"])

            # Voice TTS
            render_tts_player(result["answer"], f"msg_{len(st.session_state.messages)}")

            # Sources
            if result.get("sources"):
                with st.expander("📚 Retrieved Knowledge Sources", expanded=False):
                    for s in result["sources"][:4]:
                        st.markdown(
                            f"**{getattr(s, 'kind', 'knowledge').upper()}** | `{getattr(s, 'id', '')}` | Similarity: `{getattr(s, 'similarity', 0.8):.2f}`"
                        )
                        st.write(getattr(s, "content", "")[:300] + "...")

            # Save to session history
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "agent_trace": result.get("agent_trace", steps_captured),
                    "active_agent": result.get("active_agent", "Tech Agent"),
                    "sources": result.get("sources", []),
                    "status": result.get("status", "completed"),
                }
            )
            st.rerun()

    # Bottom disclaimer text (matching GPT layout)
    st.markdown(
        """
        <div style="text-align: center; color: #71717a; font-size: 11.5px; margin-top: 18px; margin-bottom: 4px; letter-spacing: 0.01em;">
            TechCore can make mistakes. Verify critical company IT policy information.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)




# ============================================================================
# TAB 2: Ticket Central (Search, Inspect, and Filter Company Tickets)
# ============================================================================
with tab_tickets:
    st.markdown("### 🎫 Corporate IT Ticket Central")
    st.caption("All tickets are indexed in the ChromaDB `tech_tickets` collection with cosine similarity vectors.")

    with st.expander("➕ Fast-Track Ticket Creation (Direct Submission)", expanded=False):
        tt_sub_type = st.radio("Ticket Category", ["Device Request", "Technical Incident"], horizontal=True, key="tc_cat_radio")
        if tt_sub_type == "Device Request":
            with st.form("tc_dev_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    tc_emp = st.text_input("Employee Name *", key="tc_dev_emp")
                    tc_mgr = st.text_input("Approving Manager *", key="tc_dev_mgr")
                with c2:
                    tc_dept = st.text_input("Department *", key="tc_dev_dept")
                    tc_dev = st.selectbox(
                        "Hardware",
                        [
                            'Apple MacBook Pro 14" (M3 Pro, 36GB)',
                            'Apple MacBook Air 15" (M3, 16GB)',
                            'Lenovo ThinkPad X1 Carbon Gen 12 (32GB)',
                            'Dell Precision 5690 Workstation (64GB)',
                            'Dell UltraSharp 27" 4K Monitor',
                            'CalDigit TS4 Thunderbolt Dock',
                            'Logitech MX Master 3S Combo',
                            'Jabra Evolve2 75 ANC Headset',
                        ],
                        key="tc_dev_select",
                    )
                tc_reason = st.text_area("Business Justification *", key="tc_dev_reason")
                tc_submit = st.form_submit_button("🚀 Submit Device Request")
                if tc_submit:
                    if tc_emp and tc_dept and tc_mgr and tc_reason:
                        res = agent.submit_device_request(
                            {
                                "employee": tc_emp,
                                "department": tc_dept,
                                "manager": tc_mgr,
                                "device": tc_dev,
                                "business_reason": tc_reason,
                                "work_mode": "Hybrid",
                                "preferred_date": "Standard",
                            }
                        )
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"📋 **Submitted Hardware Request:**\n"
                                    f"- **Device:** {tc_dev}\n"
                                    f"- **Employee:** {tc_emp} ({tc_dept})\n"
                                    f"- **Approver:** {tc_mgr}\n"
                                    f"- **Justification:** \"{tc_reason}\""
                                ),
                            }
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": res["answer"],
                                "agent_trace": res.get("agent_trace", []),
                                "active_agent": res.get("active_agent", "Hardware Specialist"),
                                "sources": [],
                                "status": "completed",
                            }
                        )
                        st.toast(f"Ticket dispatched: {res['ticket_id']}", icon="🎫")
                        st.rerun()
                    else:
                        st.error("Please fill in required fields marked with *.")
        else:
            with st.form("tc_iss_form", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    tc_i_emp = st.text_input("Employee Name *", key="tc_iss_emp")
                    tc_i_tag = st.text_input("Device / Asset Tag *", key="tc_iss_tag")
                with c2:
                    tc_i_urg = st.selectbox("Urgency", ["Normal", "High", "Critical"], key="tc_iss_urg")
                tc_i_sum = st.text_area("Issue Description *", key="tc_iss_sum")
                tc_i_steps = st.text_area("Troubleshooting Attempted", key="tc_iss_steps")
                tc_i_submit = st.form_submit_button("⚠️ Log Incident Ticket")
                if tc_i_submit:
                    if tc_i_emp and tc_i_tag and tc_i_sum:
                        res = agent.submit_device_issue(
                            {
                                "employee": tc_i_emp,
                                "device_asset": tc_i_tag,
                                "urgency": tc_i_urg,
                                "issue_summary": tc_i_sum,
                                "troubleshooting": tc_i_steps,
                            }
                        )
                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    f"⚠️ **Reported Technical Incident:**\n"
                                    f"- **Affected Asset:** `{tc_i_tag}`\n"
                                    f"- **Employee:** {tc_i_emp}\n"
                                    f"- **Urgency:** `{tc_i_urg.upper()}`\n"
                                    f"- **Issue Summary:** \"{tc_i_sum}\""
                                ),
                            }
                        )
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": res["answer"],
                                "agent_trace": res.get("agent_trace", []),
                                "active_agent": res.get("active_agent", "Diagnostics Specialist"),
                                "sources": [],
                                "status": "completed",
                            }
                        )
                        st.toast(f"Incident logged: {res['ticket_id']}", icon="🚨")
                        st.rerun()
                    else:
                        st.error("Please fill in required fields marked with *.")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        ticket_search = st.text_input("🔍 Search Tickets (by ID, employee name, or keyword)", placeholder="e.g. TECH-, MacBook, VPN")
    with col_t2:
        ticket_type_filter = st.selectbox("Filter Type", ["All", "device_request", "device_issue"])

    # Fetch tickets using domain tool
    ticket_res = tools.ticket_lookup_tool(agent.rag, query=ticket_search, limit=50)

    if ticket_res.success and ticket_res.output:
        all_tickets = ticket_res.output
        if ticket_type_filter != "All":
            all_tickets = [t for t in all_tickets if t.get("ticket_type") == ticket_type_filter]

        st.markdown(f"**Found {len(all_tickets)} matching tickets:**")

        for t in all_tickets:
            tid = t.get("id", "TECH-UNKNOWN")
            ttype = t.get("ticket_type", "ticket")
            doc = t.get("document", "")
            ts = t.get("timestamp", "Recent")

            with st.container():
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; color: #38bdf8; font-size: 1rem;">🎫 {tid}</span>
                            <span style="font-size: 0.78rem; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 2px 8px; border-radius: 12px; font-weight: 600;">{ttype.upper()}</span>
                        </div>
                        <p style="font-size: 0.85rem; color: #94a3b8; margin: 4px 0 8px 0;">Logged: {ts}</p>
                        <pre style="background: rgba(15, 23, 42, 0.7); padding: 8px 12px; border-radius: 6px; font-size: 0.82rem; color: #e2e8f0; white-space: pre-wrap;">{html.escape(doc)}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        st.info("No tickets recorded yet. Create a request or issue using the sidebar forms!")


# ============================================================================
# TAB 3: Memory & Knowledge Vault (Inspect Vectors, Facts & Past Logs)
# ============================================================================
with tab_memory:
    st.markdown("### 🧠 Memory & Knowledge Vault")
    st.caption("Inspect the 4 ChromaDB vector collections powering RAG context & persistent state.")

    mem_tab_policy, mem_tab_declarative, mem_tab_episodic = st.tabs(
        ["📜 Company Policy Vectors", "💡 Declarative Knowledge", "💬 Episodic Conversations"]
    )

    with mem_tab_policy:
        st.markdown("**Ingested Policy Sections from `knowledge_base/tech_department_policies.md`:**")
        try:
            policy_data = agent.rag.policy_collection.get(include=["documents", "metadatas"])
            p_ids = policy_data.get("ids", [])
            p_docs = policy_data.get("documents", [])
            p_metas = policy_data.get("metadatas", [])

            st.write(f"Total Policy Sections: **{len(p_ids)}**")
            for pid, doc, meta in zip(p_ids, p_docs, p_metas):
                title = meta.get("title", pid) if meta else pid
                with st.expander(f"📌 {title}", expanded=False):
                    st.markdown(f"**Document ID:** `{pid}`")
                    st.write(doc)
        except Exception as exc:
            st.error(f"Error loading policy collection: {exc}")

    with mem_tab_declarative:
        st.markdown("**User-Taught Facts & System Operating Rules:**")
        try:
            decl_data = agent.rag.declarative_collection.get(include=["documents", "metadatas"])
            d_ids = decl_data.get("ids", [])
            d_docs = decl_data.get("documents", [])
            d_metas = decl_data.get("metadatas", [])

            st.write(f"Total Declarative Rules: **{len(d_ids)}**")
            for did, doc, meta in zip(d_ids, d_docs, d_metas):
                src = meta.get("source", "system") if meta else "system"
                with st.expander(f"💡 [{src.upper()}] {did}", expanded=False):
                    st.write(doc)
                    if meta:
                        st.json(meta)
        except Exception as exc:
            st.error(f"Error loading declarative collection: {exc}")

    with mem_tab_episodic:
        st.markdown("**Past Interactions Auto-Saved by Memory Agent:**")
        try:
            ep_data = agent.rag.episodic_collection.get(include=["documents", "metadatas"])
            e_ids = ep_data.get("ids", [])
            e_docs = ep_data.get("documents", [])
            e_metas = ep_data.get("metadatas", [])

            st.write(f"Total Episodic Interactions: **{len(e_ids)}**")
            for eid, doc, meta in zip(reversed(e_ids), reversed(e_docs), reversed(e_metas)):
                cat = meta.get("category", "chat") if meta else "chat"
                ts = meta.get("timestamp", "recent") if meta else ""
                with st.expander(f"💬 [{cat.upper()}] {ts} — {eid}", expanded=False):
                    st.write(doc)
        except Exception as exc:
            st.error(f"Error loading episodic collection: {exc}")


# ============================================================================
# TAB 4: Diagnostic Tools Sandbox
# ============================================================================
with tab_tools:
    st.markdown("### 🛠️ IT Diagnostic Tools Sandbox")
    st.caption("Directly test the domain tools used by the LangGraph multi-agent team.")

    tool_select = st.selectbox(
        "Select Tool to Test",
        ["Hardware Catalog Lookup", "Network & VPN Diagnostics", "SLA & Escalation Calculator", "Internal Infrastructure Status"],
    )

    if tool_select == "Hardware Catalog Lookup":
        st.markdown("#### 💻 Hardware Catalog Tool")
        c1, c2 = st.columns(2)
        with c1:
            hw_cat = st.selectbox("Category", ["all", "Laptop", "Monitor", "Docking Station", "Peripherals", "Headset"])
        with c2:
            hw_max_price = st.slider("Maximum Price (USD)", min_value=100, max_value=4000, value=3000, step=100)

        hw_res = tools.hardware_catalog_tool(category=hw_cat, budget_max=hw_max_price)
        if hw_res.success:
            st.markdown(f"**Found {len(hw_res.output)} matching devices:**")
            st.dataframe(hw_res.output, use_container_width=True)
        else:
            st.error(hw_res.error)

    elif tool_select == "Network & VPN Diagnostics":
        st.markdown("#### 🌐 Network Diagnostics Tool")
        c1, c2 = st.columns(2)
        with c1:
            diag_issue = st.selectbox("Issue Category", ["vpn", "wifi", "crash", "access"])
        with c2:
            diag_os = st.selectbox("Target OS", ["windows", "macos", "linux"])

        diag_res = tools.network_diagnostics_tool(issue_type=diag_issue, os_type=diag_os)
        if diag_res.success:
            out = diag_res.output
            st.markdown(f"**Playbook:** `{out.get('title')}`")
            st.markdown(f"**Suggested Urgency:** `{out.get('suggested_priority')}` | **Team:** `{out.get('escalation_team')}`")
            st.markdown("**Recommended Remediation Steps:**")
            for s in out.get("recommended_steps", []):
                st.write(f"- {s}")
        else:
            st.error(diag_res.error)

    elif tool_select == "SLA & Escalation Calculator":
        st.markdown("#### ⏱️ SLA Resolution Calculator Tool")
        c1, c2 = st.columns(2)
        with c1:
            sla_urg = st.selectbox("Ticket Urgency", ["Normal", "High", "Critical"])
        with c2:
            sla_cat = st.selectbox("Issue Classification", ["general", "vpn", "hardware", "security", "lost_device"])

        sla_res = tools.sla_calculator_tool(urgency=sla_urg, issue_category=sla_cat)
        if sla_res.success:
            st.json(sla_res.output)
        else:
            st.error(sla_res.error)

    elif tool_select == "Internal Infrastructure Status":
        st.markdown("#### ☁️ Corporate IT Infrastructure Health")
        sys_res = tools.system_status_tool()
        if sys_res.success:
            st.dataframe(sys_res.output, use_container_width=True)
        else:
            st.error(sys_res.error)


# ============================================================================
# TAB 5: System Telemetry & Health
# ============================================================================
with tab_system:
    st.markdown("### 📊 System Telemetry & Engine Health")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    # Check Ollama
    ollama_ok = False
    try:
        ollama_ok = agent.llm.is_available()
    except Exception:
        ollama_ok = False

    with col_s1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {'#10b981' if ollama_ok else '#f59e0b'};">
                    {'ONLINE' if ollama_ok else 'OFFLINE'}
                </div>
                <div class="metric-label">Ollama LLM ({model_name})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_s2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value" style="color: #38bdf8;">ACTIVE</div>
                <div class="metric-label">LangGraph Engine</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_s3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-value" style="color: #10b981;">CONNECTED</div>
                <div class="metric-label">ChromaDB Persistent Vectors</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 🛡️ Failure & Error Handling Topology")
    st.markdown(
        """
        - **LangGraph Exception Interceptor:** Any runtime exception during agent execution is caught and routed to `node_failure_handler`.
        - **LLM Fallback Generator:** When Ollama is offline or times out, the `Responder Agent` executes deterministic structured synthesis from retrieved ChromaDB vectors.
        - **Tool Isolation:** Each domain tool runs in an isolated `try-except` boundary returning structured `ToolResult` objects without crashing the graph.
        - **Persistent Memory Protection:** ChromaDB disk operations feature idempotent deduplication slugs and fail-safe collection creation.
        """
    )


# ============================================================================
# CHATGPT-STYLE FIXED BOTTOM CHAT & VOICE INPUT PILL BAR
# ============================================================================
chat_input_val = st.chat_input(
    "Ask anything or click 🎙️ to speak...",
    accept_file=True,
    accept_audio=True,
    key="main_chat_input",
)

if chat_input_val:
    input_text = ""
    # 1. Text input from keyboard
    if hasattr(chat_input_val, "text") and chat_input_val.text:
        input_text = chat_input_val.text.strip()
    elif isinstance(chat_input_val, str) and chat_input_val.strip():
        input_text = chat_input_val.strip()

    # 2. Voice audio recorded via native microphone inside the pill bar
    if not input_text and hasattr(chat_input_val, "audio") and chat_input_val.audio:
        with st.spinner("🎙️ Transcribing voice recording..."):
            audio_text = transcribe_audio_bytes(chat_input_val.audio.getvalue())
            if audio_text:
                input_text = audio_text
            else:
                st.warning("⚠️ Could not clearly transcribe recorded audio. Please try speaking again.")

    # 3. File attachments (if uploaded via '+' button on left of chat input)
    if hasattr(chat_input_val, "files") and chat_input_val.files:
        attached_names = []
        attachment_texts = []
        for uf in chat_input_val.files:
            fname = uf.name
            attached_names.append(fname)
            try:
                # 1. Automatically store in knowledge_base/ and index in ChromaDB
                ingest_fn = getattr(agent, "ingest_document_bytes", None) or getattr(agent.rag, "ingest_document_bytes", None)
                if ingest_fn:
                    res_ing = ingest_fn(file_bytes=uf.getvalue(), filename=fname)
                    if res_ing.get("success"):
                        st.toast(f"📄 Stored in knowledge_base/ & vectorized: {fname}", icon="✅")

                # 2. Extract preview text for conversational grounding
                if fname.lower().endswith(".pdf"):
                    import io, pypdf
                    reader = pypdf.PdfReader(io.BytesIO(uf.getvalue()))
                    extracted = "\n".join([p.extract_text() or "" for p in reader.pages[:10]]).strip()
                    if extracted:
                        attachment_texts.append(f"--- Document Content: {fname} ---\n{extracted[:3500]}")
                elif fname.lower().endswith((".txt", ".md", ".log", ".json", ".csv")):
                    t_str = uf.getvalue().decode("utf-8", errors="replace").strip()
                    if t_str:
                        attachment_texts.append(f"--- Document Content: {fname} ---\n{t_str[:3500]}")
            except Exception as exc:
                logger.warning("Could not process attachment %s: %s", fname, exc)

        if not input_text:
            input_text = f"Analyze attached document(s): {', '.join(attached_names)}"

        if attachment_texts:
            input_text += "\n\n" + "\n\n".join(attachment_texts)

    if input_text:
        st.session_state.pending_prompt = input_text
        st.rerun()


