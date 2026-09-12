import os

os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import streamlit as st

from tech_agent import OllamaClient, TechDepartmentAgent, TechMemoryRAG


st.set_page_config(
    page_title="Tech Department Agent",
    page_icon="IT",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading tech agent memory...")
def load_agent(model_name: str, base_url: str, schema_version: str) -> TechDepartmentAgent:
    rag = TechMemoryRAG()
    llm = OllamaClient(model_name=model_name, base_url=base_url)
    return TechDepartmentAgent(rag=rag, llm=llm)


with st.sidebar:
    st.header("Tech Agent")
    model_name = st.text_input("Ollama model", value="mistral")
    base_url = st.text_input("Ollama URL", value="http://localhost:11434")

    agent = load_agent(model_name, base_url, "episodic-declarative-v2")
    stats = agent.rag.stats()

    st.metric("Policy sections", stats.get("policies", 0))
    st.metric("Tickets", stats.get("tickets", 0))
    st.metric("Episodic memory", stats.get("episodic_memories", 0))
    st.metric("Declarative memory", stats.get("declarative_memories", 0))

    with st.expander("Commands", expanded=False):
        st.code(
            "/help\n/policy laptop budget\n/memory vpn issue\n/remember Always ask for asset tag\n/stats\n/request\n/issue",
            language="text",
        )

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
            result = agent.teach_fact(remembered_fact.strip())
            st.success(result["answer"])
        else:
            st.warning("Add a fact or instruction before saving memory.")

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


st.title("Tech Department Support Agent")
st.caption("Policy RAG + persistent vector memory + device request and issue workflows")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi. Ask me about tech policies, device requests, software access, or device issues. Type /help for commands.",
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_message = st.chat_input("Ask the tech department agent...")

if user_message:
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy and memory..."):
            result = agent.answer(user_message)
        st.markdown(result["answer"])

        sources = result.get("sources") or []
        if sources:
            with st.expander("Retrieved context"):
                for source in sources[:6]:
                    st.markdown(
                        f"**{source.kind.title()}** | similarity `{source.similarity:.2f}` | `{source.id}`"
                    )
                    st.write(source.content)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
