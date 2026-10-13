from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = "gemini-3.8-flash"
)

prompt = PromptTemplate.from_template(
    "Answer the following question \n {question} from the following {text}"
)

parser = StrOutputParser()

url = "https://www.daraz.com.np/products/the-derma-co-4-ceramide-barrier-repair-moisturizer-niacinamide-oxylance-i-repairs-strengthens-skin-i-improves-skin-texture-i-lightweight-cream-i-24-hrs-moisturization-for-all-skin-type-50gm-i479036893-s2140059130.html?scm=1007.51610.379274.0&pvid=27855c15-463d-4793-8ddb-7ce0ccd4bb12&search=flashsale&spm=a2a0e.tm80335409.FlashSale.d_479036893"

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model |  parser

result = chain.invoke({"question":"What is the price of darma","text":"and detail before buying"})

print(result)