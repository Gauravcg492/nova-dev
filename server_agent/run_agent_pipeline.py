import os
from dotenv import load_dotenv
from server_agent.video_analysis_llm import VideoAnalysis
from server_agent.agent_baby_safety import get_baby_safety_answer

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def analyze_video_and_answer(video_path: str) -> str:
    user_question = "Is the way the baby is being cradled safe, and what should be changed?"

    video_analyzer = VideoAnalysis(api_key=OPENROUTER_API_KEY)
    video_b64 = video_analyzer.open_video_b64(video_path)
    video_report = video_analyzer.analyze_frame(video_b64)
    if not isinstance(video_report, str):
        video_report = str(video_report)

    answer = get_baby_safety_answer(video_report, user_question)
    return answer

if __name__ == "__main__":
    video_path = r"C:\Users\manig\Downloads\craddling.mp4"
    

    result = analyze_video_and_answer(video_path)
    print("FINAL ANSWER:")
    print(result)
