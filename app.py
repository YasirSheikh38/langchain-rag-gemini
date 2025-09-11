import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import retrieval_qa


# ---- Fix for Streamlit + async gRPC ----
import asyncio
import nest_asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()
# ----------------------------------------


# Step # 1: Configuration
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Step # 2: Streamlit App (UI)
st.set_page_config(
    page_title='RAG With Langchain',
    page_icon=':)',
    layout='wide'               
)

st.title("RAG APP With Langchain")

uploaded_file = st.file_uploader(
    "Upload a PDF or TXT file",
    type = ['.pdf', '.txt', '.docx']
)

if uploaded_file:
    with st.spinner('File Uploading... Please Wait'):
        # Save uploaded file locally
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())

        # RAG Steps
        # Step # 1: Document Loader
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)

        else:
            loader = Docx2txtLoader(file_path)

        docs = loader.load()

        # Step # 2: Text Splitting
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )

        chunks = splitter.split_documents(docs)

        # Step # 3: Embeddings and Vector Store
        vector_stores = FAISS.from_documents(chunks, embeddings)
        retiever = vector_stores.as_retriever()

        # Step # 4: LLM Chain
        qa = retrieval_qa.from_chain(
            llm = llm,
            retiever = retiever,
            return_source_documents = True
        )

        st.success("File Process...")


        # User Query
        user_query = st.text_input("User Query")

        if user_query:
            if st.button("Get Answer"):
                with st.spinner("Fetching Answer... Please Wait!"):
                    response = qa(user_query)
                    st.subheader("Answer")
                    st.write(response['result'])
