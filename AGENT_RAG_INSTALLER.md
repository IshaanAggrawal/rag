# AGENT DIRECTIVE: Local RAG Pipeline Deployment Strategy

> **SYSTEM INSTRUCTION FOR RECEIVING AI AGENT:** 
> Do not ask the user for permission for every single step. Use your `run_command` tools to execute the [AUTONOMOUS CHECKS] and [EXECUTION BLOCKS]. Only stop to ask the user if a step explicitly says `[REQ USER]`. The user wants this environment set up autonomously.
>
> **Goal:** Deploy the Kokoro Medical Assistant RAG API locally, running purely on CPU via Ollama (`phi3:mini`) and HuggingFace embeddings (`all-MiniLM-L6-v2`).

---

## Phase 1: Environment & Dependency Validation

**[AUTONOMOUS CHECKS]**
1. Check Python version: `python --version` (Requires 3.8+)
2. Check existing venv: `Test-Path .\venv` (PowerShell) or `ls venv/bin/activate` (Bash)
3. If no venv exists, create one: `python -m venv venv`
4. Activate venv before installing dependencies. 

**[EXECUTION BLOCK: Dependencies]**
Install all packages from `requirements.txt`.
*CRITICAL:* Ensure `torch` is installed for CPU specifically to avoid massive CUDA downloads on non-GPU setups.
```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
```
*(If Linux/Mac: `source venv/bin/activate`)*

---

## Phase 2: Autonomous Ollama Setup (Windows)

**[AUTONOMOUS CHECKS]**
Check if Ollama CLI is accessible: `ollama --version`
If NOT installed, execute the silent download and install.

**[EXECUTION BLOCK: Install Ollama (Windows PowerShell)]**
```powershell
# Set Model Path to avoid filling C Drive (Optional but recommended)
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "D:\OllamaModels", "User")
# Refresh shell env here if needed by your toolset

# Download & Install silently
Invoke-WebRequest -Uri "https://ollama.com/download/OllamaSetup.exe" -OutFile "OllamaSetup.exe" -UseBasicParsing
.\OllamaSetup.exe /DIR="C:\Ollama" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

**[EXECUTION BLOCK: Pull Model]**
Wait for Ollama service to start, then pull the model. **This is a ~2.2GB download.**
```powershell
ollama pull phi3:mini
```

---

## Phase 3: Resource Preparation

The RAG application requires `.env` variables and localized Vector Databases (Chroma) to start.

### 1. Environment Variables (`.env`)
If `.env` does not exist or is missing configs, create/append:
```env
OLLAMA_MODEL=phi3:mini
OLLAMA_BASE_URL=http://localhost:11434
# OPENAI_API_KEY is not required for this local deployment
```

### 2. Generate Vector Databases
The application explicitly looks for two Chroma DB directories (`./chroma_db` and `./gyno_db`). 

**[EXECUTION BLOCK: Create Dummy DBs]**
Run the existing script to embed sample text into Chroma. This script uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to create vector embeddings and saves them locally.
```powershell
# Ensure venv is active
python create_dummy_db.py
```
*Verification:* Check that `./chroma_db` and `./gyno_db` folders were created successfully.

---

## Phase 4: Application Spin-Up & Testing

**[EXECUTION BLOCK: Start FastAPI]**
Run `app.py` in the background or in an async wrapper so it doesn't block the agent's main thread.
```powershell
# Ensure venv is active
python app.py
```
*Note:* Uvicorn will bind to `http://0.0.0.0:8000`. Wait 15 seconds for HuggingFace embeddings to load into RAM.

**[EXECUTION BLOCK: Verify via API]**
While `app.py` is running, send an HTTP POST request to test the RAG generation.
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/rag" -Method Post -ContentType "application/json" -Body '{"message": "I have a headache", "user_id": "test_user", "role": "patient", "language": "en"}'
```
*Expected Output:* A JSON object containing `"response"` (Markdown formatted) and `"speech"` (Clean text string).

---

## Agent Troubleshooting Protocol

1. **`ModuleNotFoundError` during setup:** Ensure you explicitly activate the Virtual Environment before running `pip install` or `python` commands. Use absolute paths like `.\venv\Scripts\python.exe` if activation fails.
2. **Ollama connection refused (Port 11434):** The Ollama Windows service might not have started. Execute `start "" "C:\Ollama\ollama app.exe"` to launch it manually.
3. **Slow Inference (`> 60s`):** This is expected behavior for Phi-3 Mini strictly on CPU RAM (like an Intel Iris Xe i5).

*AGENT END DIRECTIVE*
