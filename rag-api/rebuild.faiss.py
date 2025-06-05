# rebuild_faiss.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

urls = [
    "https://www.hespress.com/%d8%b2%d9%8a%d8%a7%d8%b1%d8%a7%d8%aa-%d9%85%d9%8a%d8%af%d8%a7%d9%86%d9%8a%d8%a9-%d8%aa%d9%81%d9%82%d8%af%d9%8a%d8%a9-%d9%84%d8%a3%d8%ad%d9%8a%d8%a7%d8%a1-%d8%ac%d8%a7%d9%85%d8%b9%d9%8a%d8%a9-1571793.html", 
    "https://www.hespress.com/%d8%b3%d8%a7%d8%ad%d9%84-%d8%aa%d8%a7%d9%85%d9%88%d8%af%d8%a7-%d8%a8%d8%a7%d9%8a-%d9%8a%d8%b3%d8%aa%d8%b9%d8%af-%d9%84%d9%84%d8%b5%d9%8a%d9%81-1571726.html"
]  # Replace these with your real URLs

print("Loading documents...")
loader = UnstructuredURLLoader(urls=urls)
documents = loader.load()

print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)

print("Generating embeddings and saving FAISS store...")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_store")

print("✅ FAISS store created successfully")
