# 🤖 Techflow Chatbot (RAG)

Bem-vindo ao **Techflow Chatbot**! Este é um sistema de busca inteligente (RAG - Retrieval-Augmented Generation) que permite "conversar" com seus documentos. 

Em vez de você ler um PDF gigante procurando uma informação, você faz perguntas diretamente para ele e então retorna as respostas mais relevantes para esse contexto. O chatbot lê o conteúdo, entende o contexto e te responde usando a inteligência do LLM.

## 🚀 O que ele faz?

- **Orquestração com LangChain:** O pipeline é construído sobre o framework `LangChain`, garantindo uma integração robusta entre ingestão, busca e geração.
- **Memória de Elefante:** Ele fatia o texto em pedaços e os guarda em um banco de dados vetorial (`ChromaDB`). Quando você pergunta algo, ele busca os trechos que importam.
- **Segurança Nativa:** Inclui uma camada de sanitização de entradas para prevenir injeções e garantir o processamento seguro de dados.
- **Conversa Fluida:** Interface visual em `Streamlit` com suporte a streaming de respostas em tempo real.

## 🛠️ Como rodar?

1. **Prepare o terreno:**
   Tenha o Python instalado e instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. **Configure suas chaves:**
   Crie um arquivo `.env` (já deixei um modelo pronto) e coloque sua `GROQ_API_KEY`.

3. **Alimentção do bot:**
  O arquivo de conhecimento na pasta `data/` com o nome `techflow_base_conhecimento.pdf`.

4. **Suba o sistema:**
   Para usar a interface no navegador:
   ```
   streamlit run app.py
   ```
   Ou, se preferir o bom e velho terminal:
   ```
   python main.py
   ```

## 📂 Estrutura do Projeto

- `app.py`: A "cara" do projeto (interface Streamlit).
- `engine.py`: O "cérebro" que conecta a leitura do arquivo com a inteligência do chat.
- `core/`: Onde mora a lógica pesada de IA e banco de dados.
- `ingestion/`: Responsável por ler e picotar seus arquivos.
- `retrieval/`: Faz a busca inteligente dos trechos relevantes.

## 🛠️ Destaques Técnicos

Para elevar a qualidade e segurança do projeto, integramos as seguintes funcionalidades:

1.  **Framework de Orquestração (LangChain):** Utilizamos o LangChain para gerenciar o fluxo de dados entre o banco vetorial e o modelo de linguagem, permitindo uma arquitetura modular e escalável.
2.  **Sanitização e Segurança:** Todas as entradas do usuário passam por um rigoroso processo de limpeza (remoção de tags HTML, caracteres de controle e normalização Unicode) para prevenir ataques e garantir a integridade do sistema.

---