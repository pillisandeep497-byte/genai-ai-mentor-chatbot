from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_kEY")


)
promt = ChatPromptTemplate.from_messages([
    ("system",
    """ act as a senior generartive enginier with lots of experiance that field , like land chain , rag , gen ai , prompting 
    give advices for freshers """),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

chain = promt|llm


store = {}
def get_session(session_id:str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]

mentor = RunnableWithMessageHistory(
    chain,
    get_session,
    input_messages_key="input",
    history_messages_key="history"


)
print("======================================AI=====================================================")
print("type (exit) to quit")

while True:
    user_input = input("you: ")

    if user_input.lower() == "exit":
        break 

    response = mentor.invoke(
        {"input":user_input},
        config={
            "configurable":{
                "session_id":"sandeep"
            }
        }
    )
    print(response.text)