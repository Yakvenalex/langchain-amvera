from langchain_amvera import AmveraLLM
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = AmveraLLM(
    model="llama70b",
    temperature=0.7,
    api_token=os.getenv("AMVERA_API_TOKEN")
)

result = llm.invoke([HumanMessage(content="Привет расскажи кто ты!")])
print(result.content)
