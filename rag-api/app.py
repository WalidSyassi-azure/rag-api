import os
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY", "s3cr3t_k3y_9847")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env")

app = FastAPI(title="RAG API")

try:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
       # Skip loading FAISS now – it will be generated in a separate script
    # vectorstore = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)

    # Also comment out the qa_chain initialization since it depends on vectorstore
    # qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    #     llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY),
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever()
    # )
except Exception as e:
    raise RuntimeError(f"Failed to load retriever: {e}")

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
