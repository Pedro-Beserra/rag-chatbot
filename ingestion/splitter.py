from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Separadores que respeitam a estrutura de Markdown gerada pelo MarkItDown
_MARKDOWN_SEPARATORS = ["## ", "### ", "\n\n", "\n", " ", ""]


def split_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Divide os documentos em chunks usando separadores compatíveis
    com Markdown (headers como fronteiras naturais de corte).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_MARKDOWN_SEPARATORS,
    )
    chunks = splitter.split_documents(docs)
    print(f"[splitter] {len(docs)} documento(s) → {len(chunks)} chunks.")
    return chunks