"""
Baby Safety Monitoring System with LangChain Agents

This system uses two agents:
1. Vision Agent - Analyzes streaming video to detect baby safety issues
2. Advisory Agent - Uses memory, web search, and RAG to provide corrective actions
"""

import cv2
import base64
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from os import getenv
from dotenv import load_dotenv

load_dotenv()


class VideoStreamProcessor:
    """Handles video streaming and frame extraction"""
    
    def __init__(self, video_source: int = 0):
        """
        Initialize video stream
        Args:
            video_source: 0 for webcam, or path to video file
        """
        self.video_source = video_source
        self.cap = None
        
    def start_stream(self):
        """Start video capture"""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video source: {self.video_source}")
        print("✓ Video stream started successfully")
        
    def get_frame(self) -> Optional[bytes]:
        """Capture a single frame from video stream"""
        if self.cap is None:
            self.start_stream()
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    def frame_to_base64(self, frame_bytes: bytes) -> str:
        """Convert frame bytes to base64 string"""
        return base64.b64encode(frame_bytes).decode('utf-8')
    
    def stop_stream(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()


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

    def analyze_frame(self, frame_base64: str, frame_number: int) -> Dict[str, Any]:
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Analyze this frame (#{frame_number}) for baby safety issues. Focus on holding position, environment safety, and baby's well-being."
                },
                {
                    "type": "input_video",
                    "video_url": {
                        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                    }
                }
            ]
        )
        
        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            message
        ])
        
        return response.content


class SafetyKnowledgeBase:
    """RAG system for baby safety guidelines"""
    
    def __init__(self):
        """Initialize knowledge base with baby safety information"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create baby safety knowledge base
        safety_guidelines = """
        BABY CRADLING SAFETY GUIDELINES:
        
        1. PROPER CRADLING POSITION:
        - Always support the baby's head and neck with one hand
        - Keep the baby's head higher than their feet
        - Hold the baby close to your body for stability
        - Ensure the baby's airways are clear and face is visible
        - Never let the head fall backward or forward unsupported
        
        2. INCORRECT POSITIONS TO AVOID:
        - Cradling with head unsupported or tilted back
        - Holding baby face-down without proper support
        - Letting baby's chin touch chest (can restrict breathing)
        - One-handed holding without head support
        - Holding baby away from body (unstable)
        
        3. SAFE SLEEPING POSITIONS:
        - Always place baby on their back to sleep
        - Use firm, flat surface without pillows or soft bedding
        - Keep face and head uncovered
        - Room temperature should be comfortable
        
        4. FEEDING SAFETY:
        - Hold baby at 45-degree angle during feeding
        - Support head and neck throughout feeding
        - Keep baby's head higher than stomach
        - Burp baby in upright position
        
        5. ENVIRONMENTAL SAFETY:
        - Remove small objects that could be choking hazards
        - Keep sharp objects out of reach
        - Ensure stable surface when placing baby down
        - No loose blankets near baby's face
        - Maintain safe temperature (68-72°F)
        
        6. SIGNS OF DISTRESS:
        - Difficulty breathing or irregular breathing
        - Face turning blue or pale
        - Excessive crying or unusual quietness
        - Body stiffness or limpness
        - Seek immediate help if observed
        
        7. CORRECTIVE ACTIONS FOR IMPROPER CRADLING:
        - Gently slide hand under baby's head and neck
        - Bring baby closer to your chest
        - Adjust angle so head is elevated
        - Ensure you can see baby's face at all times
        - Use cradle hold or football hold position
        """
        
        # Split and create vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        
        texts = text_splitter.split_text(safety_guidelines)
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            collection_name="baby_safety"
        )
        
        print("✓ Safety knowledge base initialized")
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search knowledge base for relevant information"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]


class AdvisoryAgent:
    """Agent 2: Provides corrective actions using memory, web search, and RAG"""
    
    def __init__(self, api_key: str, knowledge_base: SafetyKnowledgeBase, 
                 serper_api_key: Optional[str] = None):
        """
        Initialize Advisory Agent
        
        Args:
            api_key: Anthropic API key
            knowledge_base: RAG knowledge base
            serper_api_key: Google Serper API key for web search (optional)
        """
        self.llm = ChatAnthropic(
            model="claude-sonnet-4-20250514",
            anthropic_api_key=api_key,
            max_tokens=3000
        )
        
        self.knowledge_base = knowledge_base
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup tools
        self.tools = self._setup_tools(serper_api_key)
        
        # Create agent
        self.agent = self._create_agent()
        
    def _setup_tools(self, serper_api_key: Optional[str]) -> List[Tool]:
        """Setup tools for the agent"""
        tools = []
        
        # RAG Tool
        def search_safety_guidelines(query: str) -> str:
            """Search baby safety guidelines database"""
            results = self.knowledge_base.search(query)
            return "\n\n".join(results)
        
        tools.append(Tool(
            name="search_safety_guidelines",
            func=search_safety_guidelines,
            description="Search the baby safety knowledge base for guidelines, proper techniques, and corrective actions. Use this when you need specific safety information."
        ))
        
        # Web Search Tool (if API key provided)
        if serper_api_key:
            search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
            tools.append(Tool(
                name="web_search",
                func=search.run,
                description="Search the web for latest baby safety information, pediatric guidelines, or specific safety concerns. Use when knowledge base doesn't have enough information."
            ))
        
        return tools
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a baby safety advisor AI with expertise in infant care and safety.

Your role:
1. Analyze safety issues detected from video analysis
2. Use the safety guidelines database to find relevant information
3. Search the web if needed for additional context
4. Provide clear, actionable corrective steps
5. Remember previous interactions to track recurring issues

When responding:
- Be clear and direct about the safety issue
- Provide step-by-step corrective actions
- Explain WHY the issue is dangerous
- Offer alternative safe practices
- Use empathetic but firm language
- Prioritize urgent issues

Format your response as:
⚠️ SAFETY ALERT: [Brief issue description]
🔍 ANALYSIS: [What's wrong and why it's dangerous]
✅ CORRECTIVE ACTIONS: [Step-by-step instructions]
📚 ADDITIONAL GUIDANCE: [Related safety tips]"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )
    
    def provide_advice(self, vision_analysis: Dict[str, Any]) -> str:
        """
        Generate safety advice based on vision analysis
        
        Args:
            vision_analysis: Results from vision agent
            
        Returns:
            Advisory response with corrective actions
        """
        if not vision_analysis.get("has_issue"):
            return "✓ No safety concerns detected in current frame. Continue monitoring."
        
        input_text = f"""
        Video Analysis Results (Frame #{vision_analysis['frame_number']}):
        
        {vision_analysis['raw_analysis']}
        
        Please provide:
        1. Assessment of the safety issue
        2. Specific corrective actions the caregiver should take
        3. Explanation of why this is important
        """
        
        response = self.agent.invoke({"input": input_text})
        return response["output"]


class BabySafetyMonitoringSystem:
    """Main system coordinating both agents"""
    
    def __init__(self, anthropic_api_key: str, video_source: int = 0,
                 serper_api_key: Optional[str] = None):
        """
        Initialize complete monitoring system
        
        Args:
            anthropic_api_key: Anthropic API key
            video_source: Video source (0 for webcam)
            serper_api_key: Google Serper API key (optional)
        """
        print("🚀 Initializing Baby Safety Monitoring System...")
        
        # Initialize components
        self.video_processor = VideoStreamProcessor(video_source)
        self.knowledge_base = SafetyKnowledgeBase()
        self.vision_agent = VisionAnalysisAgent(anthropic_api_key)
        self.advisory_agent = AdvisoryAgent(
            anthropic_api_key, 
            self.knowledge_base,
            serper_api_key
        )
        
        self.frame_count = 0
        self.analysis_interval = 30  # Analyze every 30 frames (approx 1 sec at 30fps)
        
        print("✓ System initialized successfully!")
    
    def process_frame(self) -> Optional[Dict[str, Any]]:
        """Process a single frame through both agents"""
        # Get frame
        frame_bytes = self.video_processor.get_frame()
        if frame_bytes is None:
            return None
        
        self.frame_count += 1
        
        # Only analyze every N frames to reduce API calls
        if self.frame_count % self.analysis_interval != 0:
            return None
        
        print(f"\n📹 Analyzing frame #{self.frame_count}...")
        
        # Agent 1: Vision Analysis
        frame_base64 = self.video_processor.frame_to_base64(frame_bytes)
        vision_analysis = self.vision_agent.analyze_frame(frame_base64, self.frame_count)
        
        print(f"Vision Analysis: {'⚠️ Issue detected' if vision_analysis['has_issue'] else '✓ Safe'}")
        
        # Agent 2: Advisory (only if issue detected)
        advisory_response = None
        if vision_analysis['has_issue']:
            print("🤖 Generating safety advice...")
            advisory_response = self.advisory_agent.provide_advice(vision_analysis)
            print(f"\n{advisory_response}\n")
        
        return {
            "frame_number": self.frame_count,
            "vision_analysis": vision_analysis,
            "advisory_response": advisory_response
        }
    
    def run_monitoring(self, duration_seconds: Optional[int] = None):
        """
        Run continuous monitoring
        
        Args:
            duration_seconds: How long to run (None for infinite)
        """
        self.video_processor.start_stream()
        print(f"\n🎥 Starting video monitoring...")
        print(f"Analyzing every {self.analysis_interval} frames\n")
        
        start_time = time.time()
        
        try:
            while True:
                # Check duration
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                # Process frame
                result = self.process_frame()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        finally:
            self.video_processor.stop_stream()
            print("✓ Video stream closed")
    
    def analyze_video_file(self, video_path: str):
        """Analyze a pre-recorded video file"""
        self.video_processor = VideoStreamProcessor(video_path)
        self.run_monitoring()


# Example usage and configuration
if __name__ == "__main__":
    # Configuration
    ANTHROPIC_API_KEY = "your-anthropic-api-key-here"  # Replace with your key
    SERPER_API_KEY = None  # Optional: Add for web search capability
    VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
    
    print("""
    ╔════════════════════════════════════════════════════╗
    ║   Baby Safety Monitoring System with AI Agents    ║
    ║                                                    ║
    ║   Agent 1: Vision Analysis (Video Understanding)  ║
    ║   Agent 2: Safety Advisory (Memory + RAG + Web)   ║
    ╚════════════════════════════════════════════════════╝
    """)
    
    # Initialize system
    try:
        system = BabySafetyMonitoringSystem(
            anthropic_api_key=ANTHROPIC_API_KEY,
            video_source=VIDEO_SOURCE,
            serper_api_key=SERPER_API_KEY
        )
        
        # Run monitoring (press Ctrl+C to stop)
        system.run_monitoring()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure:")
        print("1. You have set your ANTHROPIC_API_KEY")
        print("2. Required packages are installed")
        print("3. Camera/video source is available")