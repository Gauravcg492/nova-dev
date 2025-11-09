import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

def build_vectorstore(file_path: str, persist_path: str = "faiss_index", chunk_size: int = 500, chunk_overlap: int = 100):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension {ext}")

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ["OPENROUTER_API_KEY"]
    )

    if os.path.exists(persist_path):
        print(f"Loading vectorstore from {persist_path}")
        vectorstore = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore

    print("Building new vectorstore")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(splits, embeddings)
    vectorstore.save_local(persist_path)
    print(f"Saved vectorstore to {persist_path}")
    return vectorstore

def create_rag_tool(vectorstore, top_k: int = 4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})

    def rag_retrieve_tool(query: str) -> str:
        """Retrieve relevant document context for the given query."""
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(d.page_content for d in docs)
        return context

    return rag_retrieve_tool

    
if __name__ == "__main__":
    file_path = r"C:\Users\manig\Downloads\nova_rag.pdf"
    query = "What are the risks of swaddling?"
    vectorstore = build_vectorstore(file_path)
    tool_fn = create_rag_tool(vectorstore)
    context = tool_fn(query)
    print("Retrieved context:")
    print(context)
