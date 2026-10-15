from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class Ticket(BaseModel):
    priority:str
    angry:str

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash",
    temperature=0.2
)

struct_out = model.with_structured_output(
    Ticket
)

result = struct_out.invoke(
    "I am hurry and quite tensed and angry"
)

print(result.priority)
print(result.angry)

