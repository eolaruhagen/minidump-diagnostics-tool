import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from chonkie import SemanticChunker
from chonkie.embeddings import AutoEmbeddings
# from chonkie.cloud import SemanticChunk
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from app_logger import logger


class VectorStoreManager:
    """
    Manages vector storage for CDB command outputs.
    
    This class handles chunking and storing only CDB command outputs in a vector database.
    LLM analysis is stored separately in the citations array and not in the vector store.
    """
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.vectorstore = InMemoryVectorStore(embedding=self.embeddings)
        self.document_uuids = []  # Track all document IDs for dynamic retrieval

        embeddings = AutoEmbeddings.get_embeddings("minishlab/potion-base-8M")
        self.chunker = SemanticChunker(
            embedding_model=embeddings,
            threshold=0.8,                               
            chunk_size=2048,                             
            similarity_window=3,                         
            skip_window=0                                
        )

    def chunk_text(self, text: str) -> list[any]:
        """Chunk CDB command output text using Chonkie semantic chunking."""
        if not self.chunker:
            raise ValueError("Chonkie chunker not initialized. Cannot perform chunking.")
        
        chunks = self.chunker.chunk(text)
        return chunks
    
    def create_documents_from_command_chunks(self, 
                                           chunks: list[any],
                                           command_name: str) -> tuple[list[Document], list[str]]:
        """Create documents from CDB command output chunks."""
        documents = []
        document_uuids = []
        
        for chunk in chunks:
            doc_uuid = str(uuid.uuid4())
            documents.append(Document(
                id=doc_uuid,
                page_content=chunk.text,
                metadata={
                    "command_name": command_name,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index
                }
            ))
            document_uuids.append(doc_uuid)
        
        return documents, document_uuids
    
    def add_documents(self, documents: list[Document]) -> None:
        self.vectorstore.add_documents(documents)
        # Extract and store the document IDs
        for doc in documents:
            if hasattr(doc, 'id') and doc.id:
                self.document_uuids.append(doc.id)
    
    def get_total_document_count(self) -> int:
        """Get the total number of documents in the vector store."""
        return len(self.document_uuids)
    
    def fetch_command_text(self,
                          command_name: str,
                          document_uuids: list[str]) -> str:
        """Fetch command output text by command name and document UUIDs."""
        # Retrieve documents by their UUIDs
        retrieved_docs = self.vectorstore.get_by_ids(document_uuids)
        
        # Filter documents based on command_name
        filtered_docs = []
        for doc in retrieved_docs:
            if doc.metadata.get("command_name") == command_name:
                filtered_docs.append(doc)

        # Sort by start_index to maintain order of chunks
        filtered_docs.sort(key=lambda doc: doc.metadata.get("start_index", 0))

        # Extract text content
        command_output_texts = [doc.page_content for doc in filtered_docs]
        
        if command_output_texts:
            return f"--- Command: {command_name} ---\n" + "\n".join(command_output_texts)
        else:
            return f"--- No output found for command: {command_name} ---"
    
    def retrieve_context(self, query_text: str) -> str:
        """Retrieve relevant CDB command output context using similarity search."""
        try:
            # Use 20% of total documents as k, with min of 3 and max of 10
            total_docs = len(self.document_uuids)
            dynamic_k = max(3, min(10, int(total_docs * 0.2)))
            
            logger.info(f"Retrieving context with k={dynamic_k} from {total_docs} total documents")
            
            results = self.vectorstore.similarity_search_with_score(
                query=query_text,
                k=dynamic_k
            )
        except (AttributeError, ValueError, RuntimeError) as e:
            logger.error(f"Error during similarity search: {e}")
            return "--- No Relevant Debugging Context Retrieved ---"
        
        if not results:
            return "--- No Relevant Debugging Context Retrieved ---"
        
        # Group documents by command_name
        command_groups = {}
        for doc, score in results:
            command_name = doc.metadata.get("command_name", "unknown")
            if command_name not in command_groups:
                command_groups[command_name] = []
            
            command_groups[command_name].append((doc, score))
        
        # Build formatted output
        result_lines = ["--- Retrieved Debugging Context ---"]
        
        # Sort command groups alphabetically
        sorted_commands = sorted(command_groups.keys())
        
        for command_name in sorted_commands:
            docs_with_scores = command_groups[command_name]
            
            result_lines.append(f"\n### Command Output for {command_name}")
            # Sort by start_index to maintain chunk order
            docs_sorted = sorted(
                docs_with_scores,
                key=lambda x: x[0].metadata.get("start_index", 0)
            )
            for doc, score in docs_sorted:
                # Indent the content and add relevance score
                content_lines = doc.page_content.strip().split('\n')
                for line in content_lines:
                    result_lines.append(f"    {line}")
                result_lines.append(f"    (Relevance: {score:.4f})")
        
        return '\n'.join(result_lines)


class LLMInterface:
    """Interface for communicating with Google Generative AI LLM."""
    
    def __init__(self):
        """Initialize the LLM interface."""
        load_dotenv(override=True)
        
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY not set in .env file. Cannot connect to Google Generative AI.")
        
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            max_tokens=8000
        )
    
    def invoke(self, prompt: str) -> str:
        """
        Send prompt to LLM and get response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as a string
        """
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            logger.error(f"Error invoking LLM: {e}")
            raise


