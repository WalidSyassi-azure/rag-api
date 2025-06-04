from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Replace this with your actual content
documents = [
    Document(page_content="Render is a great platform for deployment.", metadata={"source": "intro.txt"}),
    Document(page_content="LangChain enables Retrieval-Augmented Generation.", metadata={"source": "rag.txt"})
]

# Create FAISS index
vectorstore = FAISS.from_documents(documents, embeddings)

# Save FAISS index
vectorstore.save_local("faiss_store")
print("✅ FAISS index saved to 'faiss_store/'")
