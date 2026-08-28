from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHAMADOS = [
    "A internet do laboratório caiu.",
    "Não consigo acessar minha conta.",
    "O teclado do computador parou.",
    "O Docker não inicia no Windows.",
]


def carregar_modelo() -> SentenceTransformer:
    """Carrega o modelo de embeddings usado no experimento."""
    return SentenceTransformer(MODEL_NAME)


def buscar_chamados_semelhantes(
    modelo: SentenceTransformer,
    chamados: list[str],
    novo_chamado: str,
) -> list[tuple[str, float]]:
    """
    Gera embeddings normalizados e retorna os chamados conhecidos
    ordenados pela similaridade semântica com o novo chamado.
    """
    embeddings_chamados = modelo.encode(
        chamados,
        normalize_embeddings=True,
    )

    embedding_novo = modelo.encode(
        novo_chamado,
        normalize_embeddings=True,
    )

    # Como os vetores estão normalizados, o produto escalar equivale
    # à similaridade por cosseno.
    similaridades = embeddings_chamados @ embedding_novo

    ranking = sorted(
        zip(chamados, similaridades),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    return [(texto, float(score)) for texto, score in ranking]


def main() -> None:
    print("=== Semantic HelpDesk Embeddings ===\n")

    modelo = carregar_modelo()

    novo_chamado = input(
        "Digite um novo chamado "
        "(ou pressione Enter para usar um exemplo): "
    ).strip()

    if not novo_chamado:
        novo_chamado = "Estou sem conexão com a rede"

    ranking = buscar_chamados_semelhantes(
        modelo,
        CHAMADOS,
        novo_chamado,
    )

    print(f"\nNovo chamado: {novo_chamado}\n")
    print("Ranking por similaridade semântica:")

    for posicao, (texto, score) in enumerate(ranking, start=1):
        print(f"{posicao}. {score:.4f} | {texto}")

    melhor_texto, melhor_score = ranking[0]

    print("\nChamado mais semelhante:")
    print(f"{melhor_texto}")
    print(f"Score: {melhor_score:.4f}")

    print(
        "\nObservação: o score de similaridade não representa "
        "uma probabilidade calibrada de acerto."
    )


if __name__ == "__main__":
    main()
