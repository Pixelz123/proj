import streamlit as st
import tempfile
import os
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

from src.rag_engine import create_vector_store, build_agent

st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="",
    layout="wide"
)

with st.sidebar:
    st.header("Configuration")
    
    api_key = st.text_input("Google API Key", type="password")
    
    if not api_key:
        if os.getenv("GOOGLE_API_KEY"):
            api_key = os.getenv("GOOGLE_API_KEY")
            st.success("API Key found in environment.")
        else:
            st.warning("Please enter your API Key to proceed.")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
    
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


st.title("Chat with your PDF ")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


@st.cache_resource
def setup_rag_pipeline(file_content, api_key_val):
    """
    parsing the PDF 
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_content)
        tmp_path = tmp_file.name

    try:
        vectorstore = create_vector_store(tmp_path)
        
        agent_executor = build_agent(vectorstore, api_key_val)
        
        return agent_executor
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if api_key and uploaded_file:
    try:
        with st.spinner("Processing PDF and building agent..."):
            file_content = uploaded_file.getvalue()
            agent_executor = setup_rag_pipeline(file_content, api_key)
            
        if prompt := st.chat_input("Ask about your PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                
                try:
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": st.session_state.messages},
                        config={"callbacks": [st_callback]}
                    )
                    
                    output_text = response["output"]
                    st.markdown(output_text)
                    
                    st.session_state.messages.append({"role": "assistant", "content": output_text})
                
                except Exception as e:
                    st.error(f"Error generating response: {e}")

    except Exception as e:
        st.error(f"Failed to process PDF: {e}")

elif not api_key:
    st.info("Please enter your Google API Key in the sidebar to start.")
elif not uploaded_file:
    st.info("Please upload a PDF file in the sidebar to start.")