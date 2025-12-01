import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate


@tool
def calculator(expression: str) -> str:
    """
    for baisc calculatiojn stuff
    """
    try:
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        return str(eval(expression, {"__builtins__": None}, allowed_names))
    except Exception as e:
        return f"Error calculating: {str(e)}"

def get_calculator_tool():
    return calculator


def create_vector_store(file_path: str):
    """
    for pdf :: load -> embedding and storing in vector store (in memory FAISS)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(pages)

    print("Generating embeddings locally")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

def create_rag_tool(vectorstore):
    """
    overall retrival model 
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    @tool
    def pdf_search_tool(query: str) -> str:
        """
       search tool
        """
        docs = retriever.invoke(query)
        return "\n\n".join([d.page_content for d in docs])
    
    return pdf_search_tool


def build_agent(vectorstore, google_api_key: str):
    """
    AgentExecutor for LLm + tools
    """
  
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=google_api_key,
        max_retries=2,
    )

    rag_tool = create_rag_tool(vectorstore)
    calc_tool = get_calculator_tool()
    tools = [rag_tool, calc_tool]

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

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)