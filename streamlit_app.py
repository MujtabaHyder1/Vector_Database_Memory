# ============================================================================
# streamlit_app.py — Streamlit UI for the Tech Department Support Agent
# ============================================================================
#
# This file creates the web interface for the tech agent using Streamlit.
# It provides:
#   - A chat interface for asking questions and running /commands
#   - Sidebar with live memory metrics (policy, tickets, episodic, declarative)
#   - Sidebar form to teach the agent new facts (declarative memory)
#   - Sidebar form to submit device requests (creates a ticket)
#   - Sidebar form to report device issues (creates a ticket with auto-priority)
#
# Run with: streamlit run streamlit_app.py
# Opens at: http://localhost:8501
# ============================================================================

import os

# Suppress progress bars from HuggingFace model downloads (cleaner UI)
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import streamlit as st

# Import the 3 core classes from our agent module
from tech_agent import OllamaClient, TechDepartmentAgent, TechMemoryRAG


# --- Page configuration (must be first Streamlit call) -----------------------
st.set_page_config(
    page_title="Tech Department Agent",
    page_icon="IT",
    layout="wide",  # Use full browser width
)


# --- Cache the agent so it's only loaded once (not on every Streamlit rerun) -
@st.cache_resource(show_spinner="Loading tech agent memory...")
def load_agent(model_name: str, base_url: str, schema_version: str) -> TechDepartmentAgent:
    """Initialize the RAG memory system + Ollama LLM client + Agent.
    Cached so the ChromaDB collections and embedding model are loaded only once."""
    rag = TechMemoryRAG()                                    # Load 4 ChromaDB collections + embed model
    llm = OllamaClient(model_name=model_name, base_url=base_url)  # Connect to Ollama
    return TechDepartmentAgent(rag=rag, llm=llm)


# =============================================================================
# SIDEBAR — Configuration, metrics, and forms
# =============================================================================
with st.sidebar:
    st.header("Tech Agent")

    # --- Agent configuration (model and URL) ---------------------------------
    model_name = st.text_input("Ollama model", value="mistral")
    base_url = st.text_input("Ollama URL", value="http://localhost:11434")

    # Load (or retrieve cached) agent with current settings
    agent = load_agent(model_name, base_url, "episodic-declarative-v2")
    stats = agent.rag.stats()  # Get current memory counts

    # --- Live metrics: show how many items are in each collection ------------
    st.metric("Policy sections", stats.get("policies", 0))       # From knowledge_base markdown
    st.metric("Tickets", stats.get("tickets", 0))                # Device requests + issues
    st.metric("Episodic memory", stats.get("episodic_memories", 0))  # Past conversations
    st.metric("Declarative memory", stats.get("declarative_memories", 0))  # Taught facts

    # --- Quick reference: available chat commands ----------------------------
    with st.expander("Commands", expanded=False):
        st.code(
            "/help\n/policy laptop budget\n/memory vpn issue\n/remember Always ask for asset tag\n/stats\n/request\n/issue",
            language="text",
        )

    # --- Sidebar Form 1: Teach Declarative Memory ----------------------------
    # Lets users teach the agent new facts or instructions that persist permanently
    st.divider()
    st.subheader("Teach Declarative Memory")
    with st.form("declarative_memory_form", clear_on_submit=True):
        remembered_fact = st.text_area(
            "Fact or instruction",
            placeholder="Example: Always route MacBook requests over 2000 USD to finance approval.",
        )
        submitted_memory = st.form_submit_button("Remember")

    if submitted_memory:
        if remembered_fact.strip():
            # Save the fact to declarative memory (ChromaDB collection)
            result = agent.teach_fact(remembered_fact.strip())
            st.success(result["answer"])
        else:
            st.warning("Add a fact or instruction before saving memory.")

    # --- Sidebar Form 2: Device Request ------------------------------------
    # Structured form to create a device request ticket with all required fields
    st.divider()
    st.subheader("Device Request")
    with st.form("device_request_form", clear_on_submit=True):
        req_employee = st.text_input("Employee name")
        req_department = st.text_input("Department")
        req_manager = st.text_input("Manager")
        req_device = st.selectbox(
            "Device type",
            ["Laptop", "Desktop", "Monitor", "Keyboard", "Mouse", "Headset", "Docking station", "Phone", "Charger"],
        )
        req_work_mode = st.selectbox("Work mode", ["Office", "Remote", "Hybrid"])
        req_reason = st.text_area("Business reason")
        req_date = st.date_input("Preferred date")
        submitted_request = st.form_submit_button("Create Request")

    if submitted_request:
        # Create a ticket in the tickets ChromaDB collection
        result = agent.submit_device_request(
            {
                "employee": req_employee,
                "department": req_department,
                "manager": req_manager,
                "device": req_device,
                "work_mode": req_work_mode,
                "business_reason": req_reason,
                "preferred_date": str(req_date),
            }
        )
        st.success(result["answer"])

    # --- Sidebar Form 3: Device Issue --------------------------------------
    # Structured form to report a device issue (auto-classifies priority)
    st.divider()
    st.subheader("Device Issue")
    with st.form("device_issue_form", clear_on_submit=True):
        issue_employee = st.text_input("Employee", key="issue_employee")
        issue_device = st.text_input("Device / asset tag")
        issue_urgency = st.selectbox("Urgency", ["Normal", "High", "Critical"])
        issue_summary = st.text_area("Issue summary")
        issue_steps = st.text_area("Troubleshooting tried")
        submitted_issue = st.form_submit_button("Log Issue")

    if submitted_issue:
        # Create an issue ticket — priority auto-classified from keywords
        result = agent.submit_device_issue(
            {
                "employee": issue_employee,
                "device_asset": issue_device,
                "urgency": issue_urgency,
                "issue_summary": issue_summary,
                "troubleshooting": issue_steps,
            }
        )
        st.success(result["answer"])


# =============================================================================
# MAIN CONTENT — Chat interface
# =============================================================================

st.title("Tech Department Support Agent")
st.caption("Policy RAG + persistent vector memory + device request and issue workflows")

# --- Chat history: stored in Streamlit session state -------------------------
# (This is separate from ChromaDB episodic memory — this is just for UI display)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. Ask me about tech policies, device requests, software access, or device issues. Type /help for commands.",
        }
    ]

# --- Display all previous messages in the chat ------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat input: process new user messages ----------------------------------
user_message = st.chat_input("Ask the tech department agent...")

if user_message:
    # Show the user's message in the chat
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    # Generate the agent's response (RAG search + LLM or fallback)
    with st.chat_message("assistant"):
        with st.spinner("Searching policy and memory..."):
            result = agent.answer(user_message)  # Main agent entry point
        st.markdown(result["answer"])

        # Show retrieved context sources in an expandable section
        sources = result.get("sources") or []
        if sources:
            with st.expander("Retrieved context"):
                for source in sources[:6]:
                    st.markdown(
                        f"**{source.kind.title()}** | similarity `{source.similarity:.2f}` | `{source.id}`"
                    )
                    st.write(source.content)

    # Save assistant response to chat history (for UI display)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
