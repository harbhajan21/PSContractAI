import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from phi.agent.agent import Agent
from phi.model.openai import OpenAIChat
import yfinance as yf
from dotenv import load_dotenv
import os
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Security configuration
LOGIN_CREDENTIALS = {"admin": "admin"}

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GroqAPi")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "user_question" not in st.session_state:
    st.session_state.user_question = ""


# Custom UI 
st.markdown("""
    <style>
    /* Global App Styling */
    .stApp {
        max-width: 100%;
        margin: 0;
        padding: 1rem;
        background: #f8f9fa;
        color: #333333;
        font-family: 'Inter', sans-serif;
    }

    /* Headers and Subheaders */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff;
        background: linear-gradient(90deg, #008800, #00cc00);
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.15);
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Login Container */
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 2rem;
        background: #ffffff;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        color: #333333;
    }

    /* Input Fields */
    .stTextInput input[type="text"], .stTextInput input[type="password"] {
        background: #ffffff;
        color: #333333;
        border: 1px solid #d1d5db;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        width: 100%;
        transition: border-color 0.3s, box-shadow 0.3s;
        font-size: 1rem;
    }

    /* Placeholder Text */
    .stTextInput input[type="text"]::placeholder, .stTextInput input[type="password"]::placeholder {
        color: #6b7280;
    }

    /* Hover Effect for Input Fields */
    .stTextInput input[type="text"]:hover, .stTextInput input[type="password"]:hover {
        border-color: #4b5563;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    /* Focus Effect for Input Fields */
    .stTextInput input[type="text"]:focus, .stTextInput input[type="password"]:focus {
        border-color: #008800;
        outline: none;
        box-shadow: 0 0 8px rgba(0, 136, 0, 0.3);
    }

    /* Buttons */
    .stButton>button {
        background: #008800;
        color: #ffffff;
        border-radius: 50%;
        padding: 0.6rem;
        font-size: 1.2rem;
        width: 2.5rem;
        height: 2.5rem;
        display: flex-end;
        align-items: center;
        justify-content: flex-end;
        transition: background 0.3s, transform 0.2s;
        border: none;
    }
    .stButton>button:hover {
        background: #006600;
        transform: scale(1.1);
    }

    /* Dropdowns (Selectbox) */
    .stSelectbox > div > div {
        background: #ffffff;
        color: #333333;
        border: 1px solid #d1d5db;
        border-radius: 10px;
    }
    .stSelectbox > div > div:hover {
        border-color: #4b5563;
    }

    /* Sidebar */
    .sidebar {
        background: #ffffff;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        color: #333333;
    }

    /* Metric Cards */
    .metric-card { 
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { 
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* News Cards */
    .news-card {
        background: #ffffff;
        border-left: 5px solid #008800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
        color: #333333;
    }

    /* Links */
    a {
        color: #008800;
        text-decoration: none;
    }
    a:hover {
        color: #006600;
    }

    /* Chat Messages */
    .chat-message {
        display: flex;
        margin: 0.75rem 0;
        padding: 0 1rem;
        max-width: 100%;
        align-items: flex-start;
    }
    .chat-message .avatar {
        font-size: 2rem; /* Uniform size for both user and bot avatars */
        margin-right: 0.75rem;
        align-self: flex-start;
        flex-shrink: 0;
        line-height: 1; /* Ensure consistent vertical alignment */
    }
    .chat-message .bubble {
        background: #ffffff;
        border-radius: 15px;
        padding: 1rem 1.5rem;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.1);
        font-size: 1rem;
        line-height: 1.5;
        position: relative;
        word-wrap: break-word;
        color: #333333;
        width: 100%;
    }
    .chat-message.user {
        margin-left: auto;
        flex-direction: row-reverse;
    }
    .chat-message.user .avatar {
        margin-left: 0.75rem;
        margin-right: 0;
    }
    .chat-message.user .bubble {
        background: #d1fae5;
        color: #1a4733;
    }
    .chat-message.assistant .bubble {
        background: #e5e7eb;
        color: #1f2937;
    }
    .chat-message .bubble::before {
        content: '';
        position: absolute;
        width: 0;
        height: 0;
        border: 10px solid transparent;
        top: 12px;
    }
    .chat-message.assistant .bubble::before {
        border-right-color: #e5e7eb;
        left: -20px;
    }
    .chat-message.user .bubble::before {
        border-left-color: #d1fae5;
        right: -20px;
    }
    .chat-message .timestamp {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.3rem;
        text-align: right;
        font-style: italic;
    }
    .chat-message.assistant .timestamp {
        text-align: left;
    }
    .chat-message .bubble h1, .chat-message .bubble h2, .chat-message .bubble h3 {
        background: none;
        color: #008800;
        padding: 0.5rem 0;
        box-shadow: none;
        font-size: 1.2rem;
        text-align: left;
        margin: 0.5rem 0;
    }
    .chat-message .bubble ul, .chat-message .bubble ol {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    .chat-message .bubble li {
        margin-bottom: 0.3rem;
    }

    /* Chat Input Container */
    .chat-input-container {
        display: flex;
        flex-direction: column;
        background: #ffffff;
        border: 1px solid #d1d5db;
        border-radius: 15px;
        padding: 0.75rem;
        margin: 1rem auto;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.1);
        max-width: 90%;
        transition: box-shadow 0.3s;
    }
    .chat-input-container:hover {
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
    }
    .chat-input-container .input-row {
        display: flex;
        align-items: center;
        padding: 0.25rem 0;
    }
    .chat-input-container .input-row.text-input {
        flex: 1;
        margin-bottom: 0.5rem;
    }
    .chat-input-container .stFileUploader {
        flex: 0 0 auto;
        margin-right: 1rem;
        display: inline-flex;
        align-items: center;
    }
    .chat-input-container .stFileUploader label {
        display: none;
    }
    .chat-input-container .stTextInput {
        flex: 1;
        margin: 0;
        border: none;
        box-shadow: none;
        line-height: 1.5;
    }
    .chat-input-container .stTextInput input {
        background: transparent;
        border: none;
        outline: none;
        font-size: 1rem;
        color: #333333;
        width: 100%;
    }
    .chat-input-container .stTextInput input::placeholder {
        color: #6b7280;
    }
    .chat-input-container .stButton {
        flex: 0 0 auto;
        margin-left: 1rem;
    }
    .chat-input-icon, .custom-file-upload {
        font-size: 1.5rem;
        color: #008800;
        cursor: pointer;
        margin-right: 0.5rem;
        transition: color 0.3s;
    }
    .chat-input-icon:hover, .custom-file-upload:hover {
        color: #006600;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .chat-input-container {
            padding: 0.5rem;
            max-width: 100%;
        }
        .chat-input-icon, .custom-file-upload {
            font-size: 1.3rem;
        }
        .chat-input-container .stButton>button {
            width: 2.2rem;
            height: 2.2rem;
        }
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .stApp {
            padding: 0.5rem;
        }
        .chat-message {
            max-width: 100%;
            padding: 0 0.5rem;
        }
        .chat-input-container {
            padding: 0.5rem 0.8rem;
            max-width: 100%;
        }
        .stButton>button {
            width: 2.2rem;
            height: 2.2rem;
        }
        .chat-message .bubble {
            padding: 0.8rem 1.2rem;
            font-size: 0.95rem;
        }
    }
    </style>
""", unsafe_allow_html=True)


# Authentication check
def check_login():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = None

    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.markdown("<h2 style='text-align: center; color: #2c3e50;'>Login</h2>", unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

            if submitted:
                if username in LOGIN_CREDENTIALS and password == LOGIN_CREDENTIALS[username]:
                    st.session_state.authenticated = True
                    st.session_state.agents_initialized = False
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        return False
    return True

# Initialize multi-agent for contract review
def initialize_agents():
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key", None)
    if not api_key:
        return False

    if not st.session_state.get("agents_initialized", False):
        try:
            st.session_state.clause_agent = Agent(
                name="Clause Extractor",
                role="Extracts and summarizes key clauses and terms from contracts and agreements",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "Focus on extracting important clauses such as payment terms, termination, confidentiality, liability, and deliverables.",
                    "Summarize clauses in clear, concise language.",
                    "Highlight any unusual or noteworthy terms.",
                    "Provide a structured summary of the contract's key points.",
                    "Use markdown formatting for readability.",
                    "Ensure the summary is focused on business projects and hiring agreements."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.risk_agent = Agent(
                name="Risk Analyzer",
                role="Analyzes contract for risks, ambiguous terms, and unfavorable conditions",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "Identify potential risks or ambiguous clauses.",
                    "Point out unfavorable conditions or obligations for the company.",
                    "Provide clear explanations of risks.",
                    "Use markdown formatting for readability.",
                    "Ensure the analysis is focused on business projects and hiring agreements."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.benefits_agent = Agent(
                name="Benefits Highlighter",
                role="Highlights benefits, favorable terms, and strengths in the contract",
                model=OpenAIChat(id="gpt-4o-mini"),
                instructions=[
                    "Identify and explain the positive terms and advantages for the company.",
                    "Focus on clauses that provide flexibility, protections, or beneficial deliverables.",
                    "Use markdown formatting for readability.",
                    "Ensure the analysis is focused on business projects and hiring agreements."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.contract_multi_agent = Agent(
                name="Contract Review Agent",
                role="Coordinates contract analysis by combining summaries, risks, and benefits",
                model=OpenAIChat(id="gpt-4o-mini"),
                team=[st.session_state.clause_agent, st.session_state.risk_agent, st.session_state.benefits_agent],
                instructions=[
                    "Aggregate inputs from all specialized agents.",
                    "Provide a structured report with key highlights, major points, pros, and cons.",
                    "Use markdown formatting for readability.",
                    "Focus solely on reviewing agreements, contracts, and legal documents related to business projects and hiring.",
                    "Split the report into sections: Summary, Risks, Benefits, and Key Clauses. Keep each section concise and focused. Keep Key Clauses section to a maximum of 5 key clauses with highest priority.",
                    "Sequence of sections should be: 0. Overview, 1. Key Clauses, 2. Risks, 3. Benefits, and 4. Summary.",
                    "Ensure the final report is clear, concise, and actionable.",
                    "Avoid any non-contract related topics or discussions.",
                    "Use bullet points for clarity and organization.",
                    "Provide legal advice or recommendations based on the analysis but suggest consulting a legal professional for final decisions.",
                    "If a clause is ambiguous, suggest possible interpretations or ask for clarification.",
                    "When summarizing, focus on the most relevant and impactful terms for the company."
                ],
                show_tool_calls=False,
                markdown=True
            )

            st.session_state.agents_initialized = True
            return True
        except Exception as e:
            st.error(f"Error initializing agents: {str(e)}")
            return False
    else:
        return True

def render_chat_messages():
    """Render chat history with styled user/bot messages."""
    if "chat_history" not in st.session_state:
        return

    for i, msg in enumerate(st.session_state.chat_history):
        is_user = msg["role"] == "user"
        avatar = msg.get("avatar", "🧑" if is_user else "🧑‍💻")
        bubble_class = "user" if is_user else "assistant"
        timestamp = datetime.now().strftime("%H:%M")
        #{f"<div style='float: right; font-size: 0.75rem;'>{timestamp}</div>" if is_user else ""}

        content = msg["content"].lstrip()
        st.markdown(
            f"""
            <div class="chat-message {bubble_class}">
                <span class="avatar">{avatar} </span>
                <div class="bubble">{content}</div>
            </div>
            
            """,
            unsafe_allow_html=True,
        )

def process_uploaded_file(uploaded_file):
    if uploaded_file:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("File size exceeds 5MB limit.")
            return False

        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            if not documents:
                st.error("Failed to extract content from PDF.")
                return False
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")
            return False

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        try:
            embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.get("openai_api_key"))
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.uploaded_file = uploaded_file
            st.success(f"Processed {uploaded_file.name} successfully.")
            return True
        except Exception as e:
            st.error(f"Error creating embeddings vector store: {str(e)}")
            return False
    return False


def main():
    if not check_login():
        return

    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #008800 0%, #00cc00 100%);
        color: white;
        padding: 1.5rem;
        font-size: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .chat-input-container .stButton {
        display: none; /* Hide the submit button */
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header-container">🤖 PragatiSphere - ContractAI</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.image("Logo.png", width=400)
        st.markdown('<h4>🔑 OpenAI API Key</h4>', unsafe_allow_html=True)
        api_key = st.text_input("Enter your OpenAI API Key", type="password", key="api_key_input")
        if api_key:
            st.session_state.openai_api_key = api_key
            os.environ["OPENAI_API_KEY"] = api_key
            if not st.session_state.get("agents_initialized", False):
                with st.spinner("Initializing AI agents..."):
                    initialize_agents()
        else:
            st.info("Please enter OpenAI API key to enable AI features.")

        st.markdown('<h4>📋 App Insights</h4>', unsafe_allow_html=True)
        st.markdown("""
        - **AI-Powered Contract Review**: Analyzes business contracts using specialized AI agents for clauses, risks, and benefits.
        - **Interactive Chat Interface**: Ask questions about uploaded contracts for context-based, actionable insights.
        - **Secure and User-Friendly**: Requires authentication and supports PDF uploads with a sleek, accessible UI. Not storing or save the PDFs.
        """, unsafe_allow_html=True)

    if not st.session_state.get("agents_initialized", False):
        st.warning("AI Agents not initialized. Please provide OpenAI API key.")
        return

    # Traditional file uploader (optional, kept for flexibility)
    # uploaded_file = st.file_uploader("Upload Contract / Agreement PDF (max 5MB)", type=["pdf"])
    # if uploaded_file:
    #     process_uploaded_file(uploaded_file)

    # Chat interface shown immediately after API key and agents are initialized
    # st.markdown("### Chat with Contract Document", unsafe_allow_html=True)
    render_chat_messages()

    # Initialize retrieval chain if vector store exists
    if st.session_state.get("vector_store", None):
        retriever = st.session_state.vector_store.as_retriever()
    else:
        retriever = None  # Placeholder retriever when no vector store exists

    prompt_template = ChatPromptTemplate.from_template("""
    You are a contract review AI assistant.

    Using the team of specialized agents, answer the question below with clear, concise, and structured information focusing on contract analysis.

    <context>
    {context}
    </context>

    Question: {input}

    Team:
    {{team}}
    """)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=st.session_state.get("openai_api_key"))
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retrieval_chain = create_retrieval_chain(retriever, document_chain) if retriever else None

    # Reset chat_input if submission is processed
    if "submit_trigger" not in st.session_state:
        st.session_state.submit_trigger = False
    if st.session_state.submit_trigger:
        st.session_state.chat_input = ""  # Reset before widget instantiation
        st.session_state.submit_trigger = False

    with st.container():
        st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
        st.markdown('<div class="input-row text-input">', unsafe_allow_html=True)
        uploaded_file_chat = st.file_uploader("Upload a contract PDF", type=["pdf"], key="chat_file_uploader", label_visibility="collapsed")
        if uploaded_file_chat:
            process_uploaded_file(uploaded_file_chat)
        # st.markdown('<span class="custom-file-upload">📎</span>', unsafe_allow_html=True)
        user_question = st.session_state.get("chat_input", "")  # Tie to session state
        st.text_input("Upload a contract PDF to enable context-based responses. Ask about the contract...", key="chat_input", placeholder="Type your question here...", value=user_question)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="input-row">', unsafe_allow_html=True)
        # Removed the button since Enter key will handle submission
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Handle Enter key submission
        if user_question.strip() and st.session_state.get("last_question") != user_question:
            st.session_state.last_question = user_question
            try:
                if retrieval_chain:
                    response = retrieval_chain.invoke({"input": user_question})
                    answer = response["answer"]
                else:
                    answer = "Please upload a contract PDF to enable context-based responses."
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.session_state.submit_trigger = True  # Trigger reset on next rerun
                st.experimental_set_query_params()
                st.rerun()
            except Exception as e:
                st.error(f"Error processing your question: {str(e)}")

    with st.expander("Relevant Contract Sections"):
        if st.session_state.get("chat_history") and len(st.session_state.chat_history) >= 2 and st.session_state.get("vector_store", None):
            try:
                response = retrieval_chain.invoke({"input": st.session_state.chat_history[-2]["content"]})
                for doc in response.get("context", []):
                    st.markdown(f"**Section from Page {doc.metadata.get('page', 'N/A')}:**\n{doc.page_content}", unsafe_allow_html=True)
                    st.markdown("---")
            except Exception as e:
                st.error(f"Error retrieving context: {str(e)}")

if __name__ == "__main__":
    main()


