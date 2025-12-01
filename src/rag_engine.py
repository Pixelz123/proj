import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Tool Definitions ---

@tool
def calculator(expression: str) -> str:
    """
    Useful for performing mathematical calculations. 
    Input should be a mathematical expression as a string (e.g., "200 * 5" or "10 + 5.5").
    """
    try:
        # Define allowed functions for safety
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        # Evaluate the math expression
        return str(eval(expression, {"__builtins__": None}, allowed_names))
    except Exception as e:
        return f"Error calculating: {str(e)}"

def get_calculator_tool():
    return calculator

# --- 2. RAG Pipeline Helpers ---

def create_vector_store(file_path: str):
    """
    Loads a PDF, splits it, and creates a FAISS vector store.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Split Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(pages)

    # Embed and Store
    # Note: Ensure the API key allows access to this model.
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def create_rag_tool(vectorstore):
    """
    Creates a retrieval tool bound to the specific vector store.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    @tool
    def pdf_search_tool(query: str) -> str:
        """
        Useful for searching information within the uploaded PDF document.
        Always use this tool when the user asks questions about the file content.
        """
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    
    return pdf_search_tool

# --- 3. Agent Construction ---

def build_agent(vectorstore, google_api_key: str):
    """
    Constructs the AgentExecutor with the LLM and Tools.
    """
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=google_api_key,
        max_retries=2,
    )

    # Prepare Tools
    rag_tool = create_rag_tool(vectorstore)
    calc_tool = get_calculator_tool()
    tools = [rag_tool, calc_tool]

    # Define Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "You have access to a PDF document and a calculator. "
                "Use the 'pdf_search_tool' to find information in the document. "
                "Use the 'calculator' tool for any math operations. "
                "If the user asks to calculate something based on data in the PDF, "
                "first search the PDF for the numbers, then use the calculator.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Create Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)