# HelpDesk Embeddings

Projeto experimental de classificação semântica de chamados técnicos, desenvolvido em etapas para comparar diferentes abordagens de NLP, embeddings e uso de LLMs.

O objetivo é transformar chamados escritos em linguagem natural em informações úteis para triagem, como:

- categoria provável do problema;
- localização mencionada;
- tecnologia, sistema ou dispositivo envolvido;
- descrição resumida do problema;
- prioridade;
- necessidade ou não de fallback com LLM.

---

## Evolução do projeto

Este repositório mantém duas versões do experimento para mostrar a evolução da solução.

| Versão | Abordagem | Objetivo |
|---|---|---|
| `v1-embeddings` | MiniLM + similaridade por cosseno | Explorar classificação semântica simples por embeddings |
| `v2-nlp-hybrid` | spaCy + MiniLM + regras + Qwen fallback | Estruturar o chamado, melhorar a separação entre categorias e usar LLM somente em casos ambíguos |

---

## V1 — Embeddings

A primeira versão utiliza:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- embeddings normalizados;
- similaridade por cosseno;
- ranking dos chamados de referência mais próximos.

Fluxo:

```text
Novo chamado
    ↓
Sentence Transformer
    ↓
Embedding
    ↓
Comparação com exemplos conhecidos
    ↓
Similaridade por cosseno
    ↓
Ranking semântico
```

Exemplo:

```text
Novo chamado:
Estou sem conexão com a rede
```

O sistema compara o texto com exemplos conhecidos e retorna os mais semanticamente próximos.

---

## V2 — NLP híbrido + embeddings + LLM fallback

A segunda versão mantém o embedding como principal fonte de semântica, mas adiciona uma etapa de NLP para extrair informações estruturadas.

Tecnologias principais:

- spaCy;
- `pt_core_news_sm`;
- Sentence Transformers;
- MiniLM multilíngue;
- EntityRuler;
- regras lexicais;
- Qwen 4B via Ollama como fallback.

Fluxo:

```text
                    ┌─ frase completa ─► MiniLM ─► similaridade semântica
Chamado ────────────┤
                    └─ spaCy ──────────► entidades / lemas / pistas técnicas
                                             │
                                             ▼
                                      score híbrido
                                             │
                          ┌──────────────────┴──────────────────┐
                          │                                     │
                      caso claro                           caso ambíguo
                          │                                     │
                     decisão direta                        Qwen 4B
```

O NLP pode identificar elementos como:

```text
Meu notebook não conecta no Wi-Fi do laboratório 704 desde ontem.
```

Saída estruturada aproximada:

```json
{
  "texto_normalizado": "notebook não conectar wi-fi laboratório 704 ontem",
  "local": ["laboratório 704"],
  "tecnologia": ["Wi-Fi"],
  "dispositivo": ["notebook"],
  "tempo": ["desde ontem"]
}
```

---

## Classificação híbrida

A V2 combina duas fontes de evidência:

```text
90% → similaridade semântica do embedding
10% → sinais estruturados de NLP
```

Esses pesos são experimentais e podem ser ajustados posteriormente com base em métricas reais.

O NLP não substitui a frase original antes de gerar o embedding.

Isso é importante porque o MiniLM utiliza o contexto completo da frase. A lematização e as entidades entram como informação complementar.

---

## Fallback com Qwen

O Qwen não é executado para todos os chamados.

Ele é utilizado apenas quando a classificação é considerada ambígua.

Critérios experimentais atuais:

```text
Top 1 semântico >= 0.70
→ decisão direta

Top 1 >= 0.50 e margem Top1-Top2 >= 0.15
→ decisão direta

caso contrário
→ Qwen fallback
```

O LLM recebe apenas as categorias mais prováveis e deve escolher entre elas.

Isso reduz custo computacional e evita usar um modelo generativo quando o embedding já fornece uma decisão suficientemente clara.

---

## Estrutura do repositório

```text
helpdesk-embeddings/
│
├── v1-embeddings/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── v2-nlp-hybrid/
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
```

---

## Executando a V1

Entre na pasta:

```powershell
cd v1-embeddings
```

Crie o ambiente virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Execute:

```powershell
python main.py
```

---

## Executando a V2

Entre na pasta:

```powershell
cd v2-nlp-hybrid
```

Crie o ambiente virtual:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

Confira se o Ollama está disponível:

```powershell
ollama list
```

Execute:

```powershell
python main.py
```

Para visualizar a estrutura NLP completa:

```powershell
python main.py --debug-nlp
```

---

## Ollama

A V2 utiliza por padrão:

```text
http://127.0.0.1:11434
```

com:

```text
qwen3:4b-instruct
```

`127.0.0.1` representa o próprio computador local. Publicar esse endereço no código não expõe o Ollama de quem desenvolveu o projeto.

O endereço e o modelo também podem ser configurados por variáveis de ambiente.

---

## Categorias experimentais

A versão atual trabalha com categorias como:

- Rede e Internet;
- VPN;
- Hardware;
- Acesso e Autenticação;
- Software e Aplicações;
- Banco de Dados;
- Segurança.

---

## Limitações atuais

Os scores de similaridade não representam probabilidades calibradas.

A classificação ainda utiliza uma base pequena de exemplos e regras manuais. Por isso, a qualidade precisa ser medida com um conjunto de chamados rotulados antes de afirmar ganho real de accuracy.

---

## Próximos passos

A evolução prevista inclui:

```text
V1
embeddings
    ↓
V2
NLP + embeddings + LLM fallback
    ↓
avaliação automática
    ↓
base maior de chamados rotulados
    ↓
integração com SQL
    ↓
API + interface web
```

A próxima etapa será medir:

- accuracy geral;
- accuracy por categoria;
- taxa de uso do fallback;
- matriz de confusão;
- score médio;
- margem Top1-Top2;
- casos em que o LLM melhora ou piora a classificação.

---

## Tecnologias

- Python
- spaCy
- Sentence Transformers
- MiniLM
- NLP
- Embeddings
- Similaridade por cosseno
- Ollama
- Qwen
- Git / GitHub

---

## Contexto

Este projeto foi desenvolvido como estudo de classificação semântica e processamento de linguagem natural aplicado a triagem de chamados técnicos.

A proposta é evoluir de um experimento simples com embeddings para um pipeline híbrido capaz de estruturar os chamados e recorrer a um LLM somente quando necessário.
