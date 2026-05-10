from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL


def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Instancia e retorna o modelo de embeddings.
    O modelo é multilingual e adequado para português.
    """
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)