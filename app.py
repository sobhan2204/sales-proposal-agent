import streamlit as st
from dotenv import load_dotenv
from agent import build_graph, build_refinement_graph
import logging
from io import StringIO

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Northstar Proposal Agent", layout="wide")

# Setup log capture
if "log_stream" not in st.session_state:
    st.session_state.log_stream = StringIO()
    log_handler = logging.StreamHandler(st.session_state.log_stream)
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    logging.getLogger().addHandler(log_handler)
    logging.getLogger().setLevel(logging.INFO)

st.title("🧠 Northstar AI Proposal Agent (IT Services)")
st.caption("Copilot-style agent demo • Streamlit + LangGraph + Gemini")

graph = build_graph()
refinement_graph = build_refinement_graph()

# Session memory
if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_state" not in st.session_state:
    st.session_state.last_state = None

# Sidebar: Deal selection
st.sidebar.header("📌 Demo Controls")
deal_id = st.sidebar.text_input(
    "Enter CRM Deal ID", 
    value="NS-101",
    help="Valid IDs: NS-101, NS-102 | Try an invalid ID to see error handling"
)
instruction = st.sidebar.text_area(
    "User instruction",
    value="Generate a sales proposal for this client. Keep it crisp, enterprise professional, and include scope, timeline, and pricing."
)

run = st.sidebar.button("🚀 Run Agent")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💬 Copilot Chat")
    user_msg = st.chat_input("Ask the proposal agent...")

    if user_msg:
        # Add user message to chat
        st.session_state.chat.append(("user", user_msg))
        
        # Process the message if we have a proposal state
        if st.session_state.last_state:
            with st.spinner("Agent thinking..."):
                # Update state with user feedback
                current_state = st.session_state.last_state.copy()
                current_state["user_feedback"] = user_msg
                current_state["is_initial_generation"] = False
                
                # Run refinement graph
                refined_state = refinement_graph.invoke(current_state)
                
                # Update session state
                st.session_state.last_state = refined_state
                
                # Add agent response to chat
                agent_response = refined_state.get("current_proposal", "")
                st.session_state.chat.append(("assistant", agent_response))
        else:
            # No proposal yet, ask to run agent first
            st.session_state.chat.append(("assistant", "Please click '🚀 Run Agent' first to generate an initial proposal. Then I can help you refine it!"))

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

with col2:
    st.subheader("📄 Proposal + Context")

    if run:
        initial_state = {
            "deal_id": deal_id,
            "user_instruction": instruction,
            "crm": {},
            "pricing": {},
            "template": {},
            "past": [],
            "missing_fields": [],
            "approvals": {},
            "proposal_versions": [],
            "current_proposal": "",
            "last_action": ""
        }

        with st.spinner("Agent running: gathering context → drafting → coordinating approvals..."):
            final_state = graph.invoke(initial_state)

        st.session_state.last_state = final_state
        st.success("Agent completed proposal draft + internal coordination simulation.")

    if st.session_state.last_state:
        state = st.session_state.last_state

        # Context tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Proposal Draft", "CRM Context", "Pricing", "Approvals"])

        with tab1:
            st.markdown("### ✍️ Current Proposal Draft")
            st.text_area("Proposal Output", value=state["current_proposal"], height=450)

            if state.get("proposal_versions"):
                st.caption(f"Versions: {len(state['proposal_versions'])}")

        with tab2:
            st.markdown("### 🧾 CRM Deal Context")
            st.json(state["crm"])

        with tab3:
            st.markdown("### 💰 Pricing Catalog")
            st.json(state["pricing"])

        with tab4:
            st.markdown("### ✅ Approval Tracker (Simulated Teams Workflow)")
            if state.get("approvals"):
                st.json(state["approvals"])
            else:
                st.info("No approvals requested yet.")
    else:
        st.info("Click **Run Agent** to generate a proposal draft.")

# Agent Logs Display (bottom right corner)
st.markdown("""
<style>
.log-box {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 400px;
    max-height: 250px;
    background-color: #1e1e1e;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 10px;
    overflow-y: auto;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    color: #00ff00;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    z-index: 1000;
}
.log-box::-webkit-scrollbar {
    width: 8px;
}
.log-box::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Create a container for logs
log_container = st.container()
with log_container:
    with st.expander("🔍 Agent Logs (Real-time)", expanded=False):
        log_content = st.session_state.log_stream.getvalue()
        if log_content:
            st.code(log_content, language="log")
        else:
            st.caption("No logs yet. Run the agent to see logs.")
