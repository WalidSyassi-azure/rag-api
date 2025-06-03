
import os
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY", "s3cr3t_k3y_9847")  # Custom key for API usage

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

app = FastAPI(title="RAG API")

# Load FAISS retriever
try:
    vectorstore = FAISS.load_local("faiss_store", OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
except Exception as e:
    raise RuntimeError(f"Failed to load retriever: {e}")

# Input model
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(query: Query, authorization: str = Header(None)):
    if not authorization or authorization.split(" ")[-1] != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing API key")

    try:
        result = qa_chain({"question": query.question})
        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")
