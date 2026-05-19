# 🩺 Kokoro Medical Assistant API: Local Setup Guide

Welcome to the **Kokoro Medical Assistant API**! This project utilizes FastAPI, LangGraph, and a custom RAG (Retrieval-Augmented Generation) pipeline using OpenAI & Chroma.

This guide will walk you through setting up and running the entire project completely **locally**, even on a lower-end machine (like an Intel Iris Xe laptop), without needing dedicated AWS EFS servers for vector mappings.

---

## 🛠 Prerequisites
You need the following installed on your PC:
- **Python 3.10+**
- An **OpenAI API Key** (`sk-...`)

---

## 🚀 Step-by-Step Installation

### Step 1: Set up the Virtual Environment
To keep your system clean, we use a Python Virtual Environment (`venv`). 
Open a terminal inside the project directory and run:

**For Windows (PowerShell):**
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\venv\Scripts\Activate.ps1
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

*(You should see a `(venv)` prefix in your terminal).*

---

### Step 2: Install Dependencies 
We have optimized the setup to run local embedding models natively on your CPU to avoid massive CUDA/Nvidia driver downloading if you don't have a GPU.

If you are on CPU **only**, first install standard PyTorch:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then, install the rest of the project dependencies:
```powershell
pip install fastapi uvicorn python-dotenv langchain-openai langgraph langchain-chroma langchain-community sentence-transformers emoji boto3 pydantic duckdb
```

---

### Step 3: Configure Environment Variables
The AI needs your API Key to function. 
1. Copy the `.env.example` file and rename it to `.env`.
2. Open `.env` and paste your actual API keys:

```env
OPENAI_API_KEY="sk-your-real-openai-key-here"
AWS_REGION="ap-south-1" 
```

---

### Step 4: Generate Dummy Vector Databases
This repo relies on Chroma Vector Databases (`chroma_db` and `gyno_db`). Since these are massive and not pushed to GitHub, we created a script to generate "dummy" databases so you can test the code immediately.

Run the DB generator script:
```powershell
python create_dummy_db.py
```
*This downloads a tiny 90MB embeddings model (`all-MiniLM-L6-v2`) and sets up local `./chroma_db` and `./gyno_db` folders.*

---

### Step 5: Start the API Server
Your environment is ready! Boot up the FastAPI server:

```powershell
python app.py
```
*You should see Uvicorn start successfully on `http://0.0.0.0:8000`.*

---

## 💬 How to Test the API

You do not need a frontend to test this. You can use the included `test_api.py` script.
**While your `app.py` server is running in one terminal**, open a **second terminal** and launch the test chat:

```powershell
.\venv\Scripts\Activate.ps1
python test_api.py
```

This acts as a terminal-based WhatsApp chat. You can type clinical symptoms and review both the formatted markdown response and the sanitized TTS (Text-to-Speech) strings that the API returns for frontend integration.

---

### Common Issues
**"ModuleNotFoundError" when running app.py?**
You forgot to activate your `venv`! Always run `.\venv\Scripts\Activate.ps1` before executing Python scripts in this project.

**Are my answers weird or not finding medical data?**
The databases created by `create_dummy_db.py` only contain 4-5 dummy sentences for testing the pipeline flow. For production, you must rebuild the Chroma DB vectors using real medical PDFs.
