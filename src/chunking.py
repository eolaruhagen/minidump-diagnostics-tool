from chonkie.cloud import SemanticChunker

def chunk_text(text, chonkie_api_key: str) -> list[str]:
    chunker = SemanticChunker(api_key=chonkie_api_key)
    chunks = chunker.chunk(text)
    return chunks

# def get_redis_client():
#     redis_host = os.getenv('REDIS_HOST')
#     redis_port = os.getenv('REDIS_PORT')
#     redis_password = os.getenv('REDIS_PASSWORD')
#     redis_client = Redis(host=redis_host, port=redis_port, password=redis_password)
#     return redis_client

# def push_chunks_to_redis(chunks: list[str], command_key: str):
#     redis_client = get_redis_client()
#     for chunk in chunks:
#         redis_client.lpush(command_key, chunk)