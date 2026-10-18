import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

#calling api key
load_dotenv()

#streamlit page
st.set_page_config(
    page_title = "Customer Support Bot"

)

st.title("Customer Support ChatBot")
st.write("Welcome! How can i help you!")

#initializing chatbot
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.2
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are Friendly Ashim Football Club Chatbot
        
        Football Club Information:
        - Type: Football Club
        - Traning Hours: Morning 6-9 A.M, Evening 4-7 P.M
        - Traning fee Age below 10: 3000 per month
        - Tranining fee Age above 10: 5000 per month 
        - Owner is Ashim and his sister is aashika acharya gpa b+ in class 6

         Rules:
        - Answer politely and clearly.
        - Keep answers short and helpful.
        - Use the company information provided.
        - Do not invent company policies or order details.
        - If unsure, refer the customer to support. 
        """
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human","{question}")
])

parser = StrOutputParser()

#create langchain chain
chain = prompt |llm |parser

if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

#previous message displaying from history list
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#customer input
user_input = st.chat_input("Type your question.....")

if user_input:
    #show user message
    with st.chat_message("user"):
        st.markdown(user_input)
#converting previous message to langchain format
    history = []

    for message in st.session_state.messages:
        if message["role"] ==  "user":
            history.append(
                HumanMessage(content=message["content"])
            )
        else:
            history.append(
                AIMessage(content=message["content"]) 
            )

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = chain.invoke({
                    "question": user_input,
                    "history": history
                })
                st.markdown(answer)

            except Exception:
                answer = None
                st.error(
                    "Unable to get response. Try Again."
                )


    #save message only after successful completion
    if answer is not None:
        st.session_state.messages.append({
            "role":"user",
            "content":user_input
            })

        st.session_state.messages.append({
            "role":"assistant",
            "content":"answer"
        })