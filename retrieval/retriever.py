from langchain_chroma import Chroma
from config import RETRIEVAL_K, CHUNK_SIZE

def retrieve_context(vectorstore: Chroma, query: str, k: int = RETRIEVAL_K) -> str:
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": k * 2
        }
    )

    docs = retriever.invoke(query)

    # Mantém só os melhores (reduz ruído)
    docs = docs[:k]

    context_parts = []

    for i, doc in enumerate(docs, 1):
        content = doc.page_content.strip()

        print(f"\n--- CHUNK {i} ---")
        print(content[:800])

        # 🔥 Limita tamanho por chunk
        context_parts.append(content[:CHUNK_SIZE])

    return "\n\n".join(context_parts)


def build_prompt(context: str, question: str) -> str:
    """
    Monta o prompt final que será enviado ao LLM.
    Separar a construção do prompt facilita testes e ajustes futuros.
    """
    return (
        "Responda a pergunta usando apenas o contexto fornecido.\n\n"
        "Priorize os trechos mais relevantes do contexto.\n"
        "Considere termos relacionados e sinônimos.\n"
        "Se a informação estiver parcialmente presente, responda com base no que houver.\n"
        "Não invente informações.\n"
        "Se não houver informação suficiente, diga que não encontrou no documento.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Resposta:"
    )