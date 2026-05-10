import logging
from typing import Generator, Optional

from langchain_chroma import Chroma

from core import (
    build_embeddings,
    load_vectorstore,
    create_vectorstore,
    build_client,
    complete,
    complete_stream,
    rewrite_query,
)
from ingestion import convert_to_documents, split_documents
from retrieval import retrieve_context, build_prompt
from security import sanitize_input
from config import CHROMA_PERSIST_DIR, LLM_MODEL, RETRIEVAL_K

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Orquestra todos os módulos do pipeline RAG:
    ingestão → vetorização → recuperação → geração.
    """

    def __init__(
        self,
        persist_directory: str = CHROMA_PERSIST_DIR,
        model_name: str = LLM_MODEL,
    ) -> None:
        self.persist_directory = persist_directory
        self.model_name = model_name

        self.embeddings = build_embeddings()
        self.client = build_client()
        self.vectorstore: Optional[Chroma] = load_vectorstore(
            self.embeddings, persist_directory
        )

    # ------------------------------------------------------------------
    # Ingestão
    # ------------------------------------------------------------------

    def initialize_from_file(self, file_path: str) -> None:
        """
        Converte o arquivo para Markdown, divide em chunks e cria o
        banco vetorial. Suporta PDF, DOCX, PPTX, XLSX, HTML, imagens, etc.
        Não faz nada se o banco já estiver inicializado.
        """
        if self.vectorstore is not None:
            logger.info("[engine] Banco de dados já inicializado. Ignorando.")
            return

        docs = convert_to_documents(file_path)
        chunks = split_documents(docs)
        self.vectorstore = create_vectorstore(
            chunks, self.embeddings, self.persist_directory
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _prepare_context(self, user_query: str, k: int) -> tuple[str, str, str]:
        """
        Sanitiza, reescreve a query e recupera o contexto.

        Retorna:
            clean_query     — query original sanitizada (usada no prompt final)
            rewritten_query — query expandida (usada no retrieval)
            context         — chunks recuperados concatenados
        """
        clean_query = sanitize_input(user_query)
        if not clean_query:
            raise ValueError("Query inválida ou vazia após sanitização.")

        rewritten = rewrite_query(self.client, clean_query)
        logger.debug("[engine] Query reescrita: %s", rewritten)

        context = retrieve_context(self.vectorstore, rewritten, k=k)
        return clean_query, rewritten, context

    # ------------------------------------------------------------------
    # Query — resposta completa
    # ------------------------------------------------------------------

    def query(self, user_query: str, k: int = RETRIEVAL_K) -> dict:
        """
        Sanitiza a query, faz rewrite, recupera contexto e gera resposta.

        Retorna um dict com:
            answer          — resposta do LLM
            rewritten_query — query usada no retrieval (para debug/UI)
            chunks          — trechos recuperados (para debug/UI)
        """
        if self.vectorstore is None:
            return {
                "answer": "Erro: base de conhecimento não inicializada. Chame initialize_from_file() primeiro.",
                "rewritten_query": "",
                "chunks": [],
            }

        try:
            clean_query, rewritten, context = self._prepare_context(user_query, k)
        except ValueError as e:
            return {"answer": f"Erro: {e}", "rewritten_query": "", "chunks": []}

        prompt = build_prompt(context, clean_query)
        answer = complete(self.client, prompt, model=self.model_name)

        return {
            "answer": answer,
            "rewritten_query": rewritten,
            "chunks": context.split("\n\n"),
        }

    # ------------------------------------------------------------------
    # Query — streaming real (generator)
    # ------------------------------------------------------------------

    def query_stream(
        self, user_query: str, k: int = RETRIEVAL_K
    ) -> Generator[str, None, None]:
        """
        Versão streaming de query(). Faz rewrite e retrieval de forma
        bloqueante, depois faz yield dos tokens do LLM à medida que chegam.

        Use com st.write_stream() no Streamlit:
            st.write_stream(rag.query_stream(user_input))

        Yields:
            str: fragmento de texto gerado pelo LLM.
        """
        if self.vectorstore is None:
            yield "Erro: base de conhecimento não inicializada. Chame initialize_from_file() primeiro."
            return

        try:
            clean_query, rewritten, context = self._prepare_context(user_query, k)
        except ValueError as e:
            yield f"Erro: {e}"
            return

        logger.debug("[engine] Iniciando streaming para query: %s", clean_query)
        prompt = build_prompt(context, clean_query)

        yield from complete_stream(self.client, prompt, model=self.model_name)