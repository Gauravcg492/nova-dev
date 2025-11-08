from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing import Dict, Any, Optional
import os
import base64
from dotenv import load_dotenv

load_dotenv()

class VideoAnalysis:    
    def __init__(self, api_key: str):

        self.llm = ChatOpenAI(
                        model="google/gemini-2.5-flash-preview-09-2025",  
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
        
        self.system_prompt = """You are an expert baby safety monitor analyzing video frames.
        Your responsibilities:
        1. Analyze images of babies and their caregivers
        2. Identify potential safety issues including:
        - Incorrect cradling positions (head not supported, unsafe angles)
        - Unsafe sleeping positions (face down, loose blankets)
        - Choking hazards nearby
        - Unsafe environment (sharp objects, unstable surfaces)
        - Inappropriate holding techniques
        - Signs of distress in the baby

        For each frame, provide:
        - DETECTED_ISSUE: Brief description of any safety concern (or "NONE" if safe)
        - SEVERITY: LOW, MEDIUM, HIGH, or CRITICAL
        - DESCRIPTION: Detailed observation of what you see
        - SPECIFIC_CONCERN: Exact issue that needs correction

        Be vigilant but not alarmist. Focus on actionable safety issues."""

    def open_video_b64(self, video_path):
        with open(video_path, "rb") as video_file:
           return base64.b64encode(video_file.read()).decode('utf-8')

    def analyze_frame(self, video_base64: str):
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this video for baby safety issues. Focus on holding position, environment safety, and baby's well-being."
                },
                {
                    "type": "input_video",
                    "video_url": {
                        "url": f"data:video/mp4;base64,{video_base64}"
                    }
                }
            ]
        )
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            message
        ])
        
        return response.content
    
if __name__ == "__main__":
    api_key = os.getenv("OPENROUTER_API_KEY")
    video_analyzer = VideoAnalysis(api_key=api_key)
    test_video_path = r"C:\Users\manig\Downloads\craddling.mp4" 
    result = video_analyzer.analyze_frame(video_analyzer.open_video_b64(test_video_path))
    print(f"Final Response: {result}")