from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_workflow import run_rag_pipeline
import uvicorn

app = FastAPI()

class RagRequest(BaseModel):
    message: str
    user_id: str
    language: str = "en"
    role: str = "patient"

@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    try:
        # FIX: We are now passing user_id to the pipeline!
        answer = run_rag_pipeline(
            message=req.message,
            role=req.role,
            language=req.language,
            user_id=req.user_id 
        )
        return {"response": answer}
    except Exception as e:
        return {"response": "none", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)