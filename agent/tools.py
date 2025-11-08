from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
import os

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_DEV_API")
def web_search():
    search =  GoogleSerperAPIWrapper()
    searh_tool = Tool(
                    name="Intermediate Answer",
                    func=search.run,
                    description="useful for when you need to ask with search (in internet/google)"
                )
    return searh_tool