from chonkie.cloud import SemanticChunker
from langchain_core.documents import Document

import numpy as np

def chunk_text(text, chonkie_api_key: str) -> list[SemanticChunk]:
    chunker = SemanticChunker(api_key=chonkie_api_key)
    chunks = chunker.chunk(text)
    return chunks

""" Aggregate all embeddings by sentence for a chunk given by a command"""
def aggregate_chunk_embeddings(chunks):
    # a command could have multiple chunks, each chunk has a list of sentences, which each is a list of embeddings/floats
    # all of these must be aggregated to a single vector
    embeddings = np.array([sentence.embedding for chunk in chunks for sentence in chunk.sentences])
    embeddings = embeddings.flatten()
    print(embeddings)
    return embeddings.mean(axis=0)

def flatten_chunk_embeddings(chunks: list[SemanticChunk]):
    
    