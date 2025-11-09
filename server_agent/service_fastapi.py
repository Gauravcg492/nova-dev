import os
from dotenv import load_dotenv
from video_analysis_llm import VideoAnalysis
from agent_baby_safety import get_baby_safety_answer

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def analyze_video_b64_and_answer(video_b64: str) -> str:
    video_analyzer = VideoAnalysis(api_key=OPENROUTER_API_KEY)
    video_report = video_analyzer.analyze_frame(video_b64)
    if not isinstance(video_report, str):
        video_report = str(video_report)
    user_question = "Is the way the baby is being cradled safe, and what should be changed?"
    answer = get_baby_safety_answer(video_report, user_question)
    return answer
