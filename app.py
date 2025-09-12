# Importing Library
import asyncio
import nest_asyncio
import streamlit as st
import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings



# ----------------- Logging Setup -----------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# -------------------------------------------------

# ---- Fix for Streamlit + async gRPC ----
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
nest_asyncio.apply()
# -------------------------------------------------

# Step 1: Configurations
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found! Please add it to your `.env` file.")
    logging.error("GOOGLE_API_KEY missing in .env file.")
    st.stop()

try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info("LLM and Embeddings initialized successfully.")
except Exception as e:
    st.error(f"❌ Error initializing models: {e}")
    logging.exception("Failed to initialize models")
    st.stop()

# Step 2: Streamlit App
st.set_page_config(page_title="RAG with LangChain", page_icon="🤖", layout="wide")
st.title("📂 RAG Q&A with Gemini")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "txt", "docx"])

if uploaded_file:
    try:
        with st.spinner("Processing file... Please wait ⏳"):
            # Save uploaded file locally
            file_path = f"temp_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            logging.info(f"File uploaded: {file_path}")

            # Load file
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            else:
                loader = TextLoader(file_path)

            docs = loader.load()
            if not docs:
                st.warning("⚠️ No content found in the document.")
                logging.warning("No content found in uploaded document.")
                st.stop()

            # Split text
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs)
            logging.info(f"Document split into {len(chunks)} chunks.")

            if not chunks:
                st.warning("⚠️ Document could not be split into chunks.")
                logging.warning("Failed to split document.")
                st.stop()

            # Embeddings + Vector store
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            # RAG chain
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            logging.info("RAG chain initialized successfully.")

            st.success("✅ File processed successfully!")

            # Ask a question
            user_query = st.text_input("Ask a question from your file:")
            if st.button("Get Answer"):
                if user_query.strip():
                    try:
                        with st.spinner("Fetching answer... 🤔"):
                            response = qa(user_query)

                            # Show Answer
                            st.subheader("💡 Answer")
                            st.write(response["result"])

                            # Show Metadata (Sources)
                            st.subheader("📎 Sources / Metadata")
                            for i, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {i}:** {doc.metadata}")
                                st.write(doc.page_content[:200] + "...")
                            logging.info(f"Query answered: {user_query}")
                    except Exception as e:
                        st.error(f"❌ Error while fetching answer: {e}")
                        logging.exception("Error while fetching answer")

        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Temporary file removed: {file_path}")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        logging.exception("File processing failed")
