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
    "https://www.hespress.com/%d8%aa%d9%88%d9%82%d9%8a%d9%81-%d9%85%d8%b7%d9%84%d9%88%d8%a8-%d8%a8%d8%aa%d9%87%d9%85-%d8%a7%d8%ae%d8%aa%d8%b7%d8%a7%d9%81-%d9%88%d8%a7%d8%a8%d8%aa%d8%b2%d8%a7%d8%b2-1571144.html",
    "https://www.hespress.com/%d8%a7%d9%84%d8%b9%d8%af%d9%84-%d8%aa%d9%83%d9%84%d9%81-%d8%a7%d9%84%d9%85%d8%af%d9%8a%d8%b1%d9%8a%d8%a7%d8%aa-%d8%a8%d8%b6%d8%a8%d8%b7-%d8%ad%d8%b1%d9%83%d9%8a%d8%a9-%d8%a7%d9%84%d9%85%d9%88-1571229.html"
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
