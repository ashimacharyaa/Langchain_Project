from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template
chat_template = ChatPromptTemplate([
    ('system','you are helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human','{query}')
])

chat_history=[]
#load_chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

#create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':"Where is my refund"})
print(prompt)