import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_workflow import run_rag_pipeline

app = FastAPI()

class RagRequest(BaseModel):
    message: str
    language: str = "en"  # Default English
    role: str = "patient" # Accepts 'doctor' or 'patient'

@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    try:
        # Pass language to the workflow
        answer = run_rag_pipeline(req.message, role=req.role, language=req.language)
        return {"response": answer}
    except Exception as e:
        return {"response": "none", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
