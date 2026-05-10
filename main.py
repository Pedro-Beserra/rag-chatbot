from engine import RAGEngine

def main() -> None:
    rag = RAGEngine()
    rag.initialize_from_file("./data/techflow_base_conhecimento.pdf")  # rode só uma vez

    while True:
        pergunta = input("\nDigite sua pergunta (ou 'sair'): ")
        if pergunta.lower() == "sair":
            break
        print("\n" + rag.query(pergunta))

if __name__ == "__main__":
    main()