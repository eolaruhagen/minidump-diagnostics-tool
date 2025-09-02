import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chonkie.cloud import SemanticChunker
# from chonkie.cloud import SemanticChunk
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

def chunk_text(text, chonkie_api_key: str) -> list[any]:
    """ Chunk the text into chunks """
    chunker = SemanticChunker(api_key=chonkie_api_key)
    chunks = chunker.chunk(text)
    return chunks

def create_vector_store() -> InMemoryVectorStore:
    """ I hate linter warnings """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return InMemoryVectorStore(embedding=embeddings)

def create_documents_from_command_chunks(chunks: list[any],
                                         command_name: str,
                                         is_command_output: bool) -> tuple[list[Document], list[str]]:
    """ Create documents from command chunks and return both documents and their UUIDs """
    documents = []
    document_uuids = []
    for chunk in chunks:
        doc_uuid = str(uuid.uuid4())
        documents.append(Document(
            id=doc_uuid,
            page_content=chunk.text,
            metadata={
                "command_name": command_name,
                "is_command_output": is_command_output,
                "start_index": chunk.start_index, # Assuming chunk has start_index
                "end_index": chunk.end_index # Assuming chunk has end_index
            }
        ))
        document_uuids.append(doc_uuid)
    return documents, document_uuids



def fetch_command_text(vector_store: InMemoryVectorStore,
                       command_name: str,
                       fetch_type: str,
                       document_uuids: list[str]) -> str:
    """ Fetch all text for a specified command based on fetch_type using document UUIDs. """
    # Ensure fetch_type is valid
    if fetch_type not in ["command_output", "model_analysis", "both"]:
        raise ValueError("fetch_type must be 'command_output', 'model_analysis', or 'both'")
    
    # Retrieve documents by their UUIDs
    retrieved_docs = vector_store.get_by_ids(document_uuids)
    
    # Filter documents based on command_name and fetch_type
    filtered_docs = []
    for doc in retrieved_docs:
        if doc.metadata.get("command_name") == command_name:
            if fetch_type == "both":
                filtered_docs.append(doc)
            elif fetch_type == "command_output" and doc.metadata.get("is_command_output"):
                filtered_docs.append(doc)
            elif fetch_type == "model_analysis" and not doc.metadata.get("is_command_output"):
                filtered_docs.append(doc)

    # Sort by start_index to maintain order of chunks
    filtered_docs.sort(key=lambda doc: doc.metadata.get("start_index", 0))

    command_output_texts = []
    analysis_output_texts = []

    for doc in filtered_docs:
        if doc.metadata.get("is_command_output"):
            command_output_texts.append(doc.page_content)
        else:
            analysis_output_texts.append(doc.page_content)
    
    result_string = []
    if command_output_texts:
        result_string.append("--- Command Output ---")
        result_string.append("\n".join(command_output_texts))
    
    if analysis_output_texts:
        if result_string: # Add a newline if both types are present
            result_string.append("\n")
        result_string.append("--- Model Analysis ---")
        result_string.append("\n".join(analysis_output_texts))

    return "\n".join(result_string)


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