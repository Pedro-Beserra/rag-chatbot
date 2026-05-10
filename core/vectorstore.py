import os
from typing import Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from config import CHROMA_PERSIST_DIR


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Optional[Chroma]:
    """
    Carrega um vectorstore existente, ou retorna None se não houver.
    """
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print(f"[vectorstore] Carregando banco existente em '{persist_directory}'.")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
        )
    return None


def create_vectorstore(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str = CHROMA_PERSIST_DIR,
) -> Chroma:
    """
    Cria e persiste um novo vectorstore a partir de uma lista de chunks.
    """
    print(f"[vectorstore] Criando banco com {len(chunks)} chunks em '{persist_directory}'.")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )