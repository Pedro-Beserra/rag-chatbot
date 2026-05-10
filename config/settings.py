import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM ---
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Limite de tokens na resposta do LLM.
# llama-3.1-8b-instant tem janela de ~8 192 tokens.
# Reservamos ~3 000 para prompt + contexto RAG, deixando 1 024 para a resposta.
# Aumente para 2 048 se precisar de respostas longas (ajuste o k do retriever
# proporcionalmente para não estourar a janela de contexto).
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Temperatura 0 = respostas determinísticas; suba até ~0.3 para mais fluidez
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# --- Embeddings ---
EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

# --- Vectorstore ---
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# --- Chunking ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# --- Retrieval ---
RETRIEVAL_K: int = 4  # Número de chunks recuperados por query

# --- Sanitização ---
# Tamanho máximo da query do usuário em caracteres (evita payloads gigantes)
MAX_INPUT_CHARS: int = 2000