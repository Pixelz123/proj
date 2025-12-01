import streamlit as st
import tempfile
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Import backend logic
from src.rag_engine import create_vector_store, build_agent

# --- Page Config ---
st.set_page_config(
    page_title="Gemini RAG Agent",
    page_icon="📄",
    layout="wide"
)

# --- Sidebar: Configuration ---
with st.sidebar:
    st.header("Configuration")
    
    # API Key Handling
    api_key = st.text_input("Google API Key", type="password")
    if not api_key:
        # Check env var as fallback
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("API Key found in environment.")
        else:
            st.warning("Please enter your API Key to proceed.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    # Reset button
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main Logic ---

st.title("📄 Chat with your PDF + Calculator")
st.caption("Powered by Google Gemini & LangChain")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- RAG Setup (Cached) ---

@st.cache_resource
def setup_rag_pipeline(file_content, api_key):
    """
    Processes the PDF and builds the agent. 
    Cached so we don't reload the vector store on every interaction.
    """
    # Create a temporary file to store the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    try:
        # Build Vector Store
        vectorstore = create_vector_store(tmp_path)
        
        # Build Agent
        agent_executor = build_agent(vectorstore, api_key)
        
        return agent_executor
    finally:
        # Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# --- Chat Interaction ---

if api_key and uploaded_file:
    try:
        # Build or retrieve the agent pipeline
        # We use a spinner to indicate processing
        with st.spinner("Processing PDF and building agent..."):
            file_content = uploaded_file.getvalue()
            agent_executor = setup_rag_pipeline(file_content, api_key)
            
        # Chat Input
        if prompt := st.chat_input("Ask about your PDF..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Response
            with st.chat_message("assistant"):
                # Setup callback to show the "thinking" process in Streamlit
                st_callback = StreamlitCallbackHandler(st.container())
                
                try:
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": st.session_state.messages},
                        config={"callbacks": [st_callback]}
                    )
                    
                    output_text = response["output"]
                    st.markdown(output_text)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    except Exception as e:
        st.error(f"Failed to process PDF: {e}")

elif not api_key:
    st.info("👋 Please enter your Google API Key in the sidebar to start.")
elif not uploaded_file:
    st.info("👋 Please upload a PDF file in the sidebar to start.")