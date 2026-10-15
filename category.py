from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class SupportTicket(BaseModel):
    category: str
    priority: str

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

structured_llm = llm.with_structured_output(
    SupportTicket
)

result = structured_llm.invoke(
    "My order hasn't arrived!"
)

print(result.category)
print(result.priority)