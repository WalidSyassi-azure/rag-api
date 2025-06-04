import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ⚠️ Update your URLs here or dynamically pass them
urls = [
    "https://example.com/news1",
    "https://example.com/news2"
]

# Load and split documents
loader = UnstructuredURLLoader(urls=urls)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
splits = text_splitter.split_documents(docs)

# Generate embeddings and save the FAISS index
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
db = FAISS.from_documents(splits, embeddings)
db.save_local("faiss_store")

print("✅ FAISS store rebuilt successfully!")
