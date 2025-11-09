from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from server_agent.service_fastapi import analyze_video_b64_and_answer

app = FastAPI(title="Baby Safety Video Analysis API")


class AnalyzeRequest(BaseModel):
    video_b64: str       

class AnalyzeResponse(BaseModel):
    answer: str

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        answer = analyze_video_b64_and_answer(video_b64=req.video_b64)
        return AnalyzeResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
