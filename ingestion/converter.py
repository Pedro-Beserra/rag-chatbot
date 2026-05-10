from markitdown import MarkItDown
from langchain_core.documents import Document


def convert_to_documents(file_path: str) -> list[Document]:
    """
    Converte qualquer arquivo suportado (PDF, DOCX, PPTX, XLSX,
    imagens, áudio, HTML, etc.) para um Document do LangChain
    usando MarkItDown.

    Retorna uma lista com um único Document contendo o Markdown
    extraído e os metadados de origem.
    """
    print(f"[converter] Convertendo '{file_path}' para Markdown.")
    md = MarkItDown()
    result = md.convert(file_path)

    return [
        Document(
            page_content=result.text_content,
            metadata={"source": file_path},
        )
    ]