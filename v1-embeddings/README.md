# Semantic HelpDesk Embeddings

Experimento em Python de **busca semântica aplicada a chamados de suporte**, usando Sentence Transformers e similaridade por cosseno.

## Objetivo

O projeto demonstra como representar textos como embeddings e comparar semanticamente um novo chamado com exemplos conhecidos.

A ideia surgiu durante estudos acadêmicos sobre **LLMs, embeddings e processamento de linguagem natural**.

## Como funciona

```text
Novo chamado
    ↓
Sentence Transformer
    ↓
Embedding
    ↓
Comparação com embeddings dos chamados conhecidos
    ↓
Similaridade por cosseno
    ↓
Ranking dos exemplos mais próximos
```

O modelo utilizado é:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Ele foi escolhido por oferecer suporte multilíngue, permitindo trabalhar com chamados em português.

## Estrutura

```text
semantic-helpdesk-embeddings/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalação

Crie um ambiente virtual:

```powershell
py -m venv .venv
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

## Execução

```powershell
python main.py
```

Você poderá escrever um chamado manualmente.

Exemplo:

```text
Estou sem conexão com a rede
```

O programa compara o texto com os chamados conhecidos e retorna um ranking baseado na similaridade semântica.

## Exemplos de referência

- A internet do laboratório caiu.
- Não consigo acessar minha conta.
- O teclado do computador parou.
- O Docker não inicia no Windows.

## Tecnologias

- Python
- Sentence Transformers
- Embeddings
- Similaridade por cosseno
- NLP

## Observação sobre o score

A similaridade por cosseno mede a proximidade entre os vetores gerados pelo modelo.

Um score como `0.80` não significa automaticamente que o modelo possui 80% de confiança. Para tratar o valor como probabilidade seria necessária uma etapa adicional de calibração.

## Evolução da ideia

Este repositório representa um experimento inicial de classificação semântica.

A ideia pode evoluir para:

- múltiplos exemplos por categoria;
- classificação de chamados;
- armazenamento em banco de dados;
- roteamento automático;
- fallback com LLM em casos ambíguos.
