from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    temperature=0.2
)

class ResearchResult(BaseModel):
    topic: str
    summary: str
    keywords: list[str]
    importance: int = Field(ge=1, le=10)

agent = create_agent(
    model=model,
    tools=[],
    response_format=ResearchResult
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Analyze the topic of AI in healthcare."
        }
    ]
})

print(result["structured_response"])