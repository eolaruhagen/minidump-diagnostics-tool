import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chonkie.cloud import SemanticChunker
# from chonkie.cloud import SemanticChunk
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore


class VectorStoreManager:
    """Manager class for handling vector store operations, chunking, and RAG retrieval."""
    
    def __init__(self, chonkie_api_key: str = None):
        """
        Initialize VectorStoreManager with embeddings and optional chunking capability.
        
        Args:
            chonkie_api_key: API key for Chonkie Cloud chunking (optional)
        """
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self.chonkie_api_key = chonkie_api_key
        self.chunker = SemanticChunker(api_key=chonkie_api_key) if chonkie_api_key else None
    
    def chunk_text(self, text: str) -> list[any]:
        """
        Chunk the text into semantic chunks using Chonkie Cloud.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of semantic chunks
            
        Raises:
            ValueError: If no API key was provided during initialization
        """
        if not self.chunker:
            raise ValueError("No Chonkie API key provided. Cannot perform chunking.")
        
        chunks = self.chunker.chunk(text)
        return chunks
    
    def create_documents_from_command_chunks(self, 
                                           chunks: list[any],
                                           command_name: str,
                                           is_command_output: bool) -> tuple[list[Document], list[str]]:
        """
        Create documents from command chunks and return both documents and their UUIDs.
        
        Args:
            chunks: List of semantic chunks
            command_name: Name of the command that generated the chunks
            is_command_output: True if chunks are from command output, False if from LLM analysis
            
        Returns:
            Tuple of (documents list, document UUIDs list)
        """
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
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index
                }
            ))
            document_uuids.append(doc_uuid)
        
        return documents, document_uuids
    
    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vector store."""
        self.vectorstore.add_documents(documents)
    
    def fetch_command_text(self,
                          command_name: str,
                          fetch_type: str,
                          document_uuids: list[str]) -> str:
        """
        Fetch all text for a specified command based on fetch_type using document UUIDs.
        
        Args:
            command_name: Name of the command
            fetch_type: "command_output", "model_analysis", or "both"
            document_uuids: List of document UUIDs to retrieve
            
        Returns:
            Formatted string containing the requested text
        """
        # Ensure fetch_type is valid
        if fetch_type not in ["command_output", "model_analysis", "both"]:
            raise ValueError("fetch_type must be 'command_output', 'model_analysis', or 'both'")
        
        # Retrieve documents by their UUIDs
        retrieved_docs = self.vectorstore.get_by_ids(document_uuids)
        
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
            if result_string:  # Add a newline if both types are present
                result_string.append("\n")
            result_string.append("--- Model Analysis ---")
            result_string.append("\n".join(analysis_output_texts))

        return "\n".join(result_string)
    
    def retrieve_context(self, query_text: str) -> str:
        """
        Retrieve context from vector store based on query text.
        
        Args:
            query_text: The LLM's current thought/question
            
        Returns:
            Formatted string suitable for appending to an LLM prompt
        """
        # Query vector store with similarity search
        try:
            results = self.vectorstore.similarity_search_with_score(
                query=query_text,
                k=5
            )
        except (AttributeError, ValueError, RuntimeError) as e:
            print(f"❌ Error during similarity search: {e}")
            return "--- No Relevant Debugging Context Retrieved ---"
        
        # Handle no results
        if not results:
            return "--- No Relevant Debugging Context Retrieved ---"
        
        # Group documents by command_name
        command_groups = {}
        for doc, score in results:
            command_name = doc.metadata.get("command_name", "unknown")
            if command_name not in command_groups:
                command_groups[command_name] = {
                    "command_output": [],
                    "llm_analysis": []
                }
            
            # Add document and score to appropriate group
            doc_with_score = (doc, score)
            if doc.metadata.get("is_command_output", False):
                command_groups[command_name]["command_output"].append(doc_with_score)
            else:
                command_groups[command_name]["llm_analysis"].append(doc_with_score)
        
        # Build formatted output
        result_lines = ["--- Retrieved Debugging Context ---"]
        
        # Sort command groups alphabetically
        sorted_commands = sorted(command_groups.keys())
        
        for command_name in sorted_commands:
            group = command_groups[command_name]
            
            # Process command output section
            if group["command_output"]:
                result_lines.append(f"\n### Command Output for {command_name}")
                # Sort by start_index (ascending) for command output
                command_output_sorted = sorted(
                    group["command_output"],
                    key=lambda x: x[0].metadata.get("start_index", 0)
                )
                for doc, score in command_output_sorted:
                    # Indent the content and add relevance score
                    content_lines = doc.page_content.strip().split('\n')
                    for line in content_lines:
                        result_lines.append(f"    {line}")
                    result_lines.append(f"    (Relevance: {score:.4f})")
            
            # Process LLM analysis section
            if group["llm_analysis"]:
                result_lines.append(f"\n### LLM Analysis for {command_name}")
                # Sort by relevance score (descending) for LLM analysis
                llm_analysis_sorted = sorted(
                    group["llm_analysis"],
                    key=lambda x: x[1],
                    reverse=True
                )
                for doc, score in llm_analysis_sorted:
                    # Indent the content and add relevance score
                    content_lines = doc.page_content.strip().split('\n')
                    for line in content_lines:
                        result_lines.append(f"    {line}")
                    result_lines.append(f"    (Relevance: {score:.4f})")
        
        return "\n".join(result_lines)


def test_chat_history():
    """ Test the chat history functionality """
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
    response = llm.invoke("Hello, how are you?")
    print(response)
