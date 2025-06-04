import os
import streamlit as st
import time
import base64
from dotenv import load_dotenv
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# Background styling
def set_background(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
        .stApp {{
            background: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .css-1d391kg {{
            background-color: #2e2e2e !important;
            border-right: 3px solid #555 !important;
        }}
        .stSidebar .stText {{
            font-weight: bold;
            color: white !important;
        }}
        .stTextInput label {{
            font-weight: bold !important;
            color: Black !important;
            font-size: 14px !important;
        }}
        div.stButton > button:first-child {{
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 8px;
            border: none;
        }}
        div.stButton > button:hover {{
            background-color: #45a049;
        }}
        .stTextInput > div > input {{
            border: 2px solid #76B900 !important;
            color: black !important;
            font-weight: bold;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Set background image path (adjust if deployed)
set_background("backgroundnews.jpg")

st.title("NewsBot : News Research Tool 📈")

col1, col2 = st.columns([1, 2])

with col1:
    st.sidebar.title("🔗 Enter News Article URLs")
    urls = [st.sidebar.text_input(f"URL {i+1}", placeholder="Paste URL here...") for i in range(3)]
    process_url_clicked = st.sidebar.button("📝 Process URLs")

file_path = "faiss_store"
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    with st.spinner("⏳ Processing URLs, please wait..."):
        loader = UnstructuredURLLoader(urls=[u for u in urls if u])
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        vectorstore_openai.save_local(file_path)

    st.success("✅ Data Processed Successfully!")

with col2:
    st.subheader("Ask a question regarding the websites 📰")
    query = st.text_input("🔍 Search", placeholder="What is the latest update on NVIDIA?")

    if query:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(file_path, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)

        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:8px;">
            <h4>🤖 AI Response</h4>
            <p style="font-size:18px;">{result['answer']}</p>
        </div>
        """, unsafe_allow_html=True)

        sources = result.get("sources", "")
        if sources:
            with st.expander("📚 Sources"):
                for source in sources.split("\n"):
                    st.markdown(f"✅ [{source}]({source})")

