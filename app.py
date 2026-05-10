import logging

import streamlit as st

from engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# CONFIG DA PÁGINA
# =========================
st.set_page_config(
    page_title="Techflow Chatbot",
    page_icon="☁️",
    layout="centered",
)

st.title("🤖 Techflow Chatbot")
st.caption("Faça perguntas com base no seu PDF")

# =========================
# CACHE DO RAG
# =========================
@st.cache_resource
def load_rag() -> RAGEngine:
    rag = RAGEngine()
    rag.initialize_from_file("./data/techflow_base_conhecimento.pdf")
    return rag

rag = load_rag()

# =========================
# ESTADO DA CONVERSA
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_debug" not in st.session_state:
    st.session_state.last_debug = {}

# =========================
# MOSTRAR HISTÓRICO
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# INPUT DO USUÁRIO
# =========================
user_input = st.chat_input("Digite sua pergunta...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            # query() retorna resposta completa + dados de debug.
            # Se preferir streaming real sem debug, troque por:
            #   full_response = st.write_stream(rag.query_stream(user_input))
            #   debug = {}
            result = rag.query(user_input)
            response_text = result["answer"]
            debug = {
                "rewritten_query": result["rewritten_query"],
                "chunks": result["chunks"],
            }

            # Simula streaming a partir da resposta já recebida.
            # O Groq já respondeu — isso apenas melhora a percepção de velocidade.
            def _token_generator(text: str):
                for word in text.split(" "):
                    yield word + " "

            full_response = st.write_stream(_token_generator(response_text))

        except Exception:
            logger.exception("Erro ao processar query do usuário.")
            full_response = "❌ Ocorreu um erro ao processar sua pergunta."
            debug = {}
            st.error(full_response)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    st.session_state.last_debug = debug