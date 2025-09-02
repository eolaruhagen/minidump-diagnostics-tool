import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

def test_chat_history():
    load_dotenv(override=True)

    if "GOOGLE_API_KEY" not in os.environ:
        print(
            """❌ Error: GOOGLE_API_KEY not set in .env file. 
            Cannot connect to Google Generative AI. ❌"""
            )
        return

    llm = GoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1000)
    session_id = str(uuid.uuid4())
    response = llm.invoke("Hello, how are you?")
    print(response)


def create_command_vector_store():
    return NotImplemented
    
def update_command_vector_store(command_name, command_vector):
    return NotImplemented
    
def retrieve_from_vector_store_by_command_name(command_name):
    return NotImplemented

if __name__ == "__main__":
    test_chat_history()