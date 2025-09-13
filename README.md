# 📂 RAG Q&A with Gemini using LangChain

This is a **Retrieval-Augmented Generation (RAG)** application built with **LangChain, Streamlit, and Google Gemini API**.  
It allows you to upload documents (PDF, DOCX, TXT), process them into embeddings, and then ask questions directly from the document content.

---

## 🎥 Live Demo Video
👉 Here is a live video of the working project:  
[![Watch the video](https://img.youtube.com/vi/xdbs141ZemQ/0.jpg)](https://youtu.be/xdbs141ZemQ)

---

## ✨ Features
- 📄 **Upload Files**: Supports PDF, DOCX, and TXT  
- 🧩 **Text Splitting**: Breaks documents into manageable chunks  
- 🧠 **Embeddings + FAISS Vector DB**: Stores and retrieves document chunks efficiently  
- 🤖 **LLM-Powered Q&A**: Uses Gemini model for answering queries  
- 📎 **Source Metadata**: Displays source information for transparency  
- 📝 **Logging**: Errors and processes are logged for debugging  

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
```

### 2. Create Virtual Environment & Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a .env file in the root directory and add:
```bash
GOOGLE_API_KEY = your_google_api_key
HF_API_KEY = your_huggingface_api_key
```

### 4. Run the App
```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **Python**
- **Streamlit**  
- **LangChain**  
- **Google Gemini API**  
- **FAISS**  
- **HuggingFace Embeddings**
