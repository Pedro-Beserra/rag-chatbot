import logging

from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)


def build_client() -> Groq:
    """Instancia o cliente Groq com a chave da env."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY não encontrada nas variáveis de ambiente.")
    return Groq(api_key=GROQ_API_KEY)


def complete(
    client: Groq,
    prompt: str,
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    """
    Envia o prompt ao LLM e retorna o texto da resposta completa.
    Use esta função quando não precisar de streaming (ex: rewrite_query).
    """
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def complete_stream(
    client: Groq,
    prompt: str,
    model: str = LLM_MODEL,
    max_tokens: int = LLM_MAX_TOKENS,
    temperature: float = LLM_TEMPERATURE,
):
    """
    Envia o prompt ao LLM e retorna um generator de tokens (streaming real).
    Use com st.write_stream() no Streamlit ou iterando manualmente.

    Yields:
        str: fragmento de texto à medida que o modelo o gera.
    """
    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def rewrite_query(
    client: Groq,
    question: str,
    model: str = LLM_MODEL,
) -> str:
    """
    Reescreve a pergunta do usuário para melhorar o retrieval no RAG.
    Inclui sinônimos, termos técnicos e variações relevantes.

    A query reescrita é usada apenas no retrieval — a pergunta original
    é mantida no prompt final enviado ao LLM.
    """
    prompt = (
        "Reescreva a pergunta do usuário para melhorar a busca em um sistema RAG.\n"
        "Inclua sinônimos, termos técnicos e variações relevantes.\n"
        "Não responda a pergunta, apenas reescreva.\n\n"
        f"Pergunta original: {question}\n\n"
        "Pergunta reescrita:"
    )

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        max_tokens=128,
        temperature=0.0,
    )

    return response.choices[0].message.content.strip()