# 🚀 Deploying Kokoro Medical Bot to Render

You can deploy this application to **Render** using one of two methods. 

---

## ⚡ Method 1: Single-Service Standalone Deployment (Recommended & Simplest)
Since we implemented **Direct Pipeline Mode**, the Streamlit frontend can run the LangGraph workflow directly in its own process, querying the local Chroma vector database and calling the Groq cloud API. This means **you do not need to run or deploy a separate FastAPI backend!**

This method is highly recommended on Render's free tier because it avoids cold starts, connection errors, and hosting costs of running two services.

### Step-by-Step Instructions:
1. Push all your code changes to your GitHub repository:
   ```bash
   git add .
   git commit -m "Configure Groq and Streamlit frontend"
   git push origin main
   ```
2. Log in to [Render](https://render.com/) and click **New +** → **Web Service**.
3. Connect your GitHub repository: `https://github.com/IshaanAggrawal/rag.git`
4. Configure the Web Service settings:
   - **Name**: `kokoro-medical-bot`
   - **Environment**: `Docker`
   - **Branch**: `main`
   - **Docker Command**: `streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0`
5. Click **Advanced** to add **Environment Variables**:
   - `LLM_PROVIDER`: `groq`
   - `GROQ_API_KEY`: `gsk_your_groq_api_key_here`
   - `GROQ_MODEL`: `llama-3.3-70b-versatile`
6. Click **Deploy Web Service**.
7. Once deployed, open the Render URL (e.g., `https://kokoro-medical-bot.onrender.com`).
8. **In the sidebar, check "Direct Pipeline Mode (Bypass FastAPI)"** to run it completely standalone!

---

## 🏢 Method 2: Two-Service Deployment (Production Standard)
If you require a strict separation of frontend and backend (e.g., if you plan to connect mobile apps or other clients to the FastAPI backend), you can deploy them as two separate Render services.

### Service A: FastAPI Backend
1. Click **New +** → **Web Service** on Render.
2. Select your GitHub repository.
3. Configure the settings:
   - **Name**: `kokoro-backend`
   - **Environment**: `Docker`
   - **Docker Command**: `python app.py`
4. Add **Environment Variables**:
   - `LLM_PROVIDER`: `groq`
   - `GROQ_API_KEY`: `gsk_your_groq_api_key_here`
   - `GROQ_MODEL`: `llama-3.3-70b-versatile`
5. Click **Deploy Web Service** and copy its generated URL (e.g., `https://kokoro-backend.onrender.com`).

### Service B: Streamlit Frontend
1. Click **New +** → **Web Service** on Render.
2. Select your GitHub repository.
3. Configure the settings:
   - **Name**: `kokoro-frontend`
   - **Environment**: `Docker`
   - **Docker Command**: `streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0`
4. Click **Advanced** and add the following **Environment Variables**:
   - `LLM_PROVIDER`: `groq`
   - `GROQ_API_KEY`: `gsk_your_groq_api_key_here`
   - `GROQ_MODEL`: `llama-3.3-70b-versatile`
5. Deploy the service.
6. Open your Streamlit application and in the sidebar:
   - Set the **Backend API URL** field to your deployed backend URL: `https://kokoro-backend.onrender.com`
   - Uncheck **Direct Pipeline Mode** to use the API mode!
