
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Load Gemini API key from .env
load_dotenv()

# 2. Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0.3,
)

# 3. Create customer support instructions
system_prompt = """
You are a friendly customer support assistant for ABC Store.

Company policies:
- Delivery usually takes 3-5 business days.
- Customers can return items within 7 days.
- Support email: support@example.com
- Business hours: Monday-Friday, 9 AM-6 PM.

Rules:
1. Answer politely and clearly.
2. Keep answers short and helpful.
3. Answer in the customer's language.
4. If information is unavailable, say you don't know.
5. Never invent order status or customer details.
6. For order-specific issues, refer customers to support.
"""

# 4. Build LangChain prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# 5. Create LangChain chain
chain = prompt | llm

# 6. Streamlit page setup
st.set_page_config(
    page_title="Customer Support Bot",
)

st.title(" ABC Store Customer Support")


# 7. Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 8. Clear chat button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# 9. Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 10. Accept customer question
user_input = st.chat_input("Ask me anything...")

if user_input:

    # Display customer's message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Convert previous chats to LangChain messages
    history = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    # Save customer's latest message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate assistant reply
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "history": history,
                    "question": user_input
                })

                answer = response.text
                st.markdown(answer)

            # Save assistant's answer
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Something went wrong: {e}")
