from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv


load_dotenv()
api_key = "GEMINI_API_KEY"

model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature = 0.2
)

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Product price")
    category: str = Field(description="Product category")

prompt = ChatPromptTemplate.from_template(
    "Extract product information from: {text}"
)

structured_llm = model.with_structured_output(Product)

chain = prompt | structured_llm

result = chain.invoke({
    "text": "Samsung Galaxy phone costs $850, "
            "category Electronics."
})

print(result.model_dump())