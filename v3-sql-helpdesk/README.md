# AI HelpDesk SQL

Sistema local de triagem de chamados técnicos que combina NLP, embeddings, fallback com LLM, SQLite, SQLAlchemy, FastAPI e uma interface web.

## Arquitetura

```text
Chamado
   │
   ├── spaCy
   │     ├── local
   │     ├── tecnologia
   │     ├── dispositivo
   │     ├── sistema
   │     └── lemas relevantes
   │
   ├── MiniLM
   │     └── similaridade com exemplos do SQLite
   │
   └── score híbrido
          │
          ├── caso claro ── decisão direta
          │
          └── caso ambíguo ── Qwen 4B
                                  │
                                  ├── chamado original
                                  ├── NLP estruturado
                                  └── Top 3 categorias
          │
          ▼
     roteamento
          │
          ├── departamento
          ├── analista com menor fila
          └── quantidade de chamados à frente
          │
          ▼
     SQLite + resposta ao usuário
```

## Resposta ao usuário

Depois da classificação e do roteamento, a aplicação gera uma confirmação usando dados reais do banco, por exemplo:

```text
Seu problema de acesso ao Docker no laboratório 704 foi enviado ao departamento de Rede e Internet.
Rafael ficará responsável pelo atendimento. Atualmente há 2 chamados na sua frente com esse responsável.
```

O nome do responsável, departamento e tamanho da fila não são inventados pelo LLM. Eles vêm do SQLite.

## Revisão humana da IA

A interface possui uma aba `Revisão IA` para validar as previsões uma a uma. O revisor pode marcar a classificação como correta ou selecionar a categoria correta e registrar uma observação.

O feedback é armazenado em `ai_feedback` e pode ser analisado pela view `v_ai_quality`. A revisão não altera retroativamente o roteamento do chamado; ela registra a verdade humana para avaliação e evolução do classificador.

A fila de revisão exibe somente previsões ainda sem feedback humano e mostra os scores semântico, NLP, híbrido, margem, uso do Qwen e informações estruturadas extraídas pelo spaCy.

## Chamados abertos

A interface possui uma tela específica para chamados em aberto. Cada chamado pode ser marcado como concluído.

Ao concluir:

- o status passa para `RESOLVED`;
- `resolved_at` é preenchido;
- o evento é registrado em `ticket_history`;
- o chamado deixa de aparecer na lista de abertos;
- os dados continuam salvos no banco para histórico e métricas.

## Estrutura

```text
v3-sql-helpdesk/
├── app/
│   ├── services/
│   │   ├── nlp_service.py
│   │   ├── ollama_client.py
│   │   ├── semantic_classifier.py
│   │   ├── ticket_ai.py
│   │   └── ticket_service.py
│   ├── config.py
│   ├── database.py
│   ├── db_setup.py
│   ├── models.py
│   ├── repositories.py
│   ├── schemas.py
│   └── web.py
├── database/
│   ├── schema.sql
│   ├── seed.sql
│   └── useful_queries.sql
├── data/
│   └── .gitkeep
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── .env.example
├── .gitignore
├── SECURITY.md
├── check_models.py
├── init_db.py
├── main.py
├── requirements.txt
└── README.md
```

## Instalação

```powershell
cd v3-sql-helpdesk

py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m spacy download pt_core_news_sm

Copy-Item .env.example .env
python check_models.py
python init_db.py
python main.py
```

Abra:

```text
http://127.0.0.1:8000
```

Documentação FastAPI:

```text
http://127.0.0.1:8000/docs
```

## Modelos

Embedding:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Fallback:

```text
qwen3:4b-instruct
```

O Qwen é chamado somente quando o classificador híbrido considera o chamado ambíguo.

## Banco de dados

O SQLite é criado localmente em:

```text
data/helpdesk.db
```

Esse arquivo não faz parte do repositório.

As categorias, palavras-chave de NLP e exemplos usados pelos embeddings ficam no próprio banco, permitindo inspeção e evolução sem alterar o classificador principal.


## Segurança

- Configurações locais são carregadas pelo arquivo `.env`, que não é versionado.
- Bancos SQLite gerados localmente também são ignorados pelo Git.
- As consultas da aplicação utilizam SQLAlchemy e parâmetros vinculados, evitando concatenação direta de entradas do usuário em SQL.
- FastAPI e Ollama são configurados para execução local por padrão.
