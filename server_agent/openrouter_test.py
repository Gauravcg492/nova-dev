from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from os import getenv
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm = ChatOpenAI(
    model="google/gemini-2.5-flash-preview-09-2025",  
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

# print(llm_chain.run(question))

message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": "What is in this image, audio and video?"
        },
        {
            "type": "input_video",
            "video_url": {
                "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            }
        }
    ]
)

response = llm.invoke([message])
print(response.content)