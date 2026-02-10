from fastapi import FastAPI
from pydantic import BaseModel
from langgraph_workflow import run_rag_pipeline
import uvicorn
from utils import clean_text_for_speech  # <--- 1. Ye naya import add kiya

app = FastAPI()

class RagRequest(BaseModel):
    message: str
    user_id: str
    language: str = "en"
    role: str = "patient"

@app.post("/rag")
async def rag_endpoint(req: RagRequest):
    try:
        # 1. Original Jawab (Markdown + Emojis wala)
        answer = run_rag_pipeline(
            message=req.message,
            role=req.role,
            language=req.language,
            user_id=req.user_id 
        )
        
        # 2. Audio ke liye safai (Clean Text)
        clean_audio_text = clean_text_for_speech(answer)

        # 3. Dono wapas bhejo
        return {
            "response": answer,           # Screen ke liye (Formatting ke saath)
            "speech": clean_audio_text    # Bolne ke liye (Ekdum saaf)
        }

    except Exception as e:
        return {"response": "Error occurred", "speech": "Error occurred", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)