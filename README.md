# HelpDesk Embeddings

Projeto de classificação e triagem inteligente de chamados técnicos, desenvolvido em etapas para explorar a evolução de uma solução baseada em **NLP, embeddings, LLMs e banco de dados**.

O projeto começou como um experimento simples de similaridade semântica e evoluiu para uma aplicação web capaz de receber chamados, classificá-los, encaminhá-los automaticamente e registrar feedback humano sobre as decisões da IA.

---

## Objetivo

Transformar chamados escritos em linguagem natural em informações úteis para triagem, como:

- categoria provável do problema;
- localização mencionada;
- tecnologia, sistema ou dispositivo envolvido;
- descrição resumida do problema;
- prioridade;
- departamento responsável;
- funcionário responsável;
- necessidade ou não de fallback com LLM;
- registro de feedback humano sobre a classificação.

---

# Evolução do projeto

O repositório mantém diferentes versões para mostrar a evolução da solução e as decisões tomadas em cada etapa.

| Versão | Abordagem | Objetivo |
|---|---|---|
| `v1-embeddings` | MiniLM + similaridade por cosseno | Explorar classificação semântica simples por embeddings |
| `v2-nlp-hybrid` | spaCy + MiniLM + regras + Qwen fallback | Combinar semântica com informações estruturadas e utilizar LLM apenas em casos ambíguos |
| `v3-sql-helpdesk` | NLP + embeddings + Qwen + SQLite + FastAPI | Transformar o classificador em uma aplicação completa de triagem e roteamento de chamados |

---

# V1 — Embeddings

A primeira versão utiliza:

- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`;
- embeddings normalizados;
- similaridade por cosseno;
- ranking dos exemplos conhecidos mais próximos.

### Fluxo

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

O sistema transforma a frase em um vetor e compara esse embedding com os embeddings de chamados conhecidos.

As frases semanticamente mais próximas recebem os maiores scores.

---

# V2 — NLP híbrido + embeddings + LLM fallback

A segunda versão mantém o embedding como principal fonte de informação semântica, mas adiciona uma etapa paralela de NLP.

Tecnologias principais:

- spaCy;
- `pt_core_news_sm`;
- Sentence Transformers;
- MiniLM multilíngue;
- EntityRuler;
- regras lexicais;
- Qwen 4B executado localmente através do Ollama.

### Arquitetura

```text
                    ┌─ frase completa ─► MiniLM ─► similaridade semântica
Chamado ────────────┤
                    │
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

O spaCy pode identificar informações como:

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

Esses pesos são experimentais.

O NLP **não substitui a frase original utilizada pelo MiniLM**.

O fluxo é:

```text
Frase original
     │
     ├──────────────► MiniLM
     │                  │
     │                  └── embedding da frase completa
     │
     └──────────────► spaCy
                        │
                        └── entidades, lemas e pistas técnicas
```

Isso preserva o contexto completo utilizado pelo Transformer enquanto permite adicionar informações estruturadas ao classificador.

---

## Fallback com Qwen

O Qwen não é executado para todos os chamados.

Ele é utilizado somente quando a classificação é considerada ambígua.

Critérios experimentais atuais:

```text
Top 1 semântico >= 0.70
→ decisão direta

Top 1 >= 0.50
e margem Top1 - Top2 >= 0.15
→ decisão direta

caso contrário
→ Qwen fallback
```

Nos casos ambíguos, o Qwen recebe:

```text
texto original
+
informações estruturadas pelo spaCy
+
Top 3 categorias mais prováveis
```

O modelo deve escolher entre essas categorias.

Essa arquitetura evita executar um modelo generativo quando o classificador semântico já possui evidências suficientes para tomar uma decisão.

---

# V3 — AI HelpDesk SQL

A terceira versão transforma o experimento de classificação em uma aplicação completa de HelpDesk.

A V3 adiciona:

- SQLite;
- SQLAlchemy;
- FastAPI;
- interface web;
- persistência dos chamados;
- departamentos;
- funcionários;
- distribuição automática de chamados;
- fila de atendimento;
- histórico;
- registro das decisões da IA;
- revisão humana das classificações.

### Arquitetura

```text
USUÁRIO
  │
  │ cria chamado
  ▼
FastAPI
  │
  ▼
SQLite
  │
  └── ticket criado como OPEN
  │
  ▼
spaCy NLP
  │
  ├── local
  ├── tecnologia
  ├── dispositivo
  ├── sistema
  ├── tempo
  └── lemas relevantes
  │
  ▼
MiniLM
  │
  ├── busca exemplos no SQLite
  ├── gera embedding da frase completa
  ├── compara com classification_examples
  └── gera ranking semântico
  │
  ▼
NLP + embedding
  │
  └── score híbrido
  │
  ├──────── CASO CLARO
  │              │
  │              └── classificação direta
  │
  └──────── CASO AMBÍGUO
                 │
                 ▼
              Qwen 4B
                 │
                 ├── texto original
                 ├── NLP estruturado
                 └── Top 3 categorias
                 │
                 ▼
            categoria final
                 │
                 ▼
              SQLite
                 │
                 ├── categoria → departamento
                 ├── funcionário com menor fila
                 └── quantidade de chamados à frente
                 │
                 ▼
       previsão + histórico
                 │
                 ▼
              FastAPI
                 │
                 ▼
            navegador
```

---

## Roteamento automático

Depois que a categoria é definida, o sistema utiliza o banco de dados para descobrir qual departamento deve receber o chamado.

Em seguida, procura o funcionário daquele departamento com a menor quantidade de chamados abertos.

O usuário recebe uma resposta baseada nos dados reais armazenados no SQLite.

Exemplo:

```text
Seu problema de acesso ao Docker foi enviado ao departamento responsável.

Rafael ficará responsável pelo atendimento.

Atualmente há 2 chamados na sua frente com esse responsável.
```

Departamento, funcionário e tamanho da fila não são gerados pelo LLM.

Essas informações vêm do banco de dados.

---

## Ciclo de vida dos chamados

Chamados podem permanecer em estados como:

```text
OPEN
TRIAGED
IN_PROGRESS
RESOLVED
```

Quando um chamado é marcado como concluído:

```text
status → RESOLVED
resolved_at → preenchido
histórico → registrado
```

O chamado deixa de aparecer na fila de chamados abertos, mas permanece no banco para histórico e análise.

---

# Revisão humana da IA

A V3 adiciona um fluxo de **Human-in-the-Loop**.

A interface possui uma área dedicada à revisão das classificações realizadas pelo sistema.

Para cada previsão, o revisor pode visualizar informações como:

```text
chamado
categoria prevista
score semântico
score NLP
score híbrido
margem entre categorias
uso ou não do Qwen
origem da decisão
entidades extraídas
```

O revisor pode então indicar:

```text
✓ classificação correta
```

ou:

```text
✕ classificação incorreta
        │
        ▼
categoria correta
        │
        ▼
feedback salvo
```

Os resultados são armazenados em:

```text
ai_feedback
```

Esse mecanismo permite criar posteriormente métricas reais de qualidade do classificador.

O feedback humano atualmente é registrado para avaliação e análise; ele não realiza treinamento automático do modelo.

---

# Banco de dados

A V3 utiliza SQLite.

O banco é criado localmente em:

```text
data/helpdesk.db
```

Entre as principais estruturas estão:

```text
departments
employees
categories
classification_examples
category_keywords
tickets
ai_predictions
ai_feedback
ticket_history
```

Também existem views para facilitar análise e acompanhamento dos dados.

Exemplos:

```text
v_ticket_overview
v_open_ticket_queue
v_ai_quality
```

Os exemplos semânticos utilizados pelo MiniLM também ficam armazenados no SQLite.

```text
classification_examples
        │
        ▼
MiniLM
        │
        ▼
embeddings
        │
        ▼
comparação com novo chamado
```

As palavras e pesos utilizados como evidência adicional pelo NLP ficam em:

```text
category_keywords
```

Isso permite alterar parte do comportamento do classificador sem precisar modificar diretamente o código Python.

---

# Segurança

A aplicação foi projetada para execução local.

- Configurações locais são carregadas pelo arquivo `.env`, que não é versionado.
- Bancos SQLite gerados localmente também são ignorados pelo Git.
- As consultas da aplicação utilizam SQLAlchemy e parâmetros vinculados, evitando concatenação direta de entradas do usuário em SQL.
- FastAPI e Ollama são configurados para execução local por padrão.

O arquivo `.env.example` contém apenas configurações públicas necessárias para configurar o ambiente.

---

# Estrutura do repositório

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
├── v3-sql-helpdesk/
│   ├── app/
│   │   ├── services/
│   │   │   ├── nlp_service.py
│   │   │   ├── ollama_client.py
│   │   │   ├── semantic_classifier.py
│   │   │   ├── ticket_ai.py
│   │   │   └── ticket_service.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── db_setup.py
│   │   ├── models.py
│   │   ├── repositories.py
│   │   ├── schemas.py
│   │   └── web.py
│   │
│   ├── database/
│   │   ├── schema.sql
│   │   ├── seed.sql
│   │   └── useful_queries.sql
│   │
│   ├── data/
│   │   └── .gitkeep
│   │
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   │
│   ├── .env.example
│   ├── .gitignore
│   ├── check_models.py
│   ├── init_db.py
│   ├── main.py
│   ├── requirements.txt
│   └── README.md
│
├── .gitignore
└── README.md
```

---

# Executando a V1

Entre na pasta:

```powershell
cd v1-embeddings
```

Crie e ative o ambiente virtual:

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

# Executando a V2

Entre na pasta:

```powershell
cd v2-nlp-hybrid
```

Crie e ative o ambiente virtual:

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

Para visualizar informações adicionais do NLP:

```powershell
python main.py --debug-nlp
```

---

# Executando a V3

Entre na pasta:

```powershell
cd v3-sql-helpdesk
```

Crie o ambiente virtual:

```powershell
py -m venv .venv
```

No PowerShell, caso a execução de scripts esteja bloqueada:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```powershell
python -m pip install -r requirements.txt
```

Instale o modelo do spaCy:

```powershell
python -m spacy download pt_core_news_sm
```

Crie o arquivo local de configuração:

```powershell
Copy-Item .env.example .env
```

Confira os modelos necessários:

```powershell
python check_models.py
```

Inicialize o banco:

```powershell
python init_db.py
```

Execute a aplicação:

```powershell
python main.py
```

A interface estará disponível em:

```text
http://127.0.0.1:8000
```

A documentação automática da API estará disponível em:

```text
http://127.0.0.1:8000/docs
```

---

# Ollama

As versões que utilizam LLM usam Ollama localmente.

Por padrão:

```text
http://127.0.0.1:11434
```

Modelo utilizado como fallback:

```text
qwen3:4b-instruct
```

O Qwen é utilizado somente quando os scores e a margem indicam que o classificador não possui confiança suficiente para tomar uma decisão direta.

---

# Categorias experimentais

Atualmente o projeto trabalha com categorias como:

```text
Rede e Internet
VPN
Hardware
Acesso e Autenticação
Software e Aplicações
Banco de Dados
Segurança
```

As categorias e seus exemplos podem ser expandidos através do banco de dados.

---

# Métricas e avaliação

A V3 foi preparada para permitir uma avaliação mais objetiva das decisões da IA através do registro de previsões e feedback humano.

Entre as métricas que podem ser calculadas estão:

```text
accuracy geral
accuracy por categoria
taxa de uso do fallback
matriz de confusão
score médio
margem média Top1 - Top2
quantidade de correções humanas
casos em que o Qwen melhora a classificação
casos em que o Qwen piora a classificação
```

Essa etapa é importante porque os scores de similaridade do MiniLM **não representam probabilidades calibradas**.

Um score de:

```text
0.82
```

representa alta similaridade semântica dentro daquele espaço vetorial, mas não significa necessariamente:

```text
82% de probabilidade de estar correto
```

---

# Limitações atuais

O projeto ainda utiliza uma base relativamente pequena de exemplos semânticos e regras lexicais.

Os thresholds e pesos do classificador híbrido são experimentais e ainda precisam ser avaliados utilizando um conjunto maior de chamados rotulados.

O feedback humano já pode ser armazenado, mas ainda não é utilizado automaticamente para atualizar os exemplos de classificação ou treinar novos modelos.

O sistema também foi desenvolvido para execução local e não possui, nesta versão, autenticação de usuários ou infraestrutura destinada à exposição pública da API.

---

# Próximos passos

A evolução prevista agora está concentrada principalmente na avaliação e melhoria quantitativa do sistema:

```text
V1
Embeddings
    ↓
V2
NLP + embeddings + LLM fallback
    ↓
V3
HelpDesk + SQL + API + interface web
    ↓
dataset maior de chamados rotulados
    ↓
avaliação automática
    ↓
matriz de confusão e análise de erros
    ↓
ajuste dos thresholds e pesos
    ↓
uso do feedback humano para evolução do classificador
```

Também podem ser exploradas futuramente:

```text
testes automatizados
autenticação
controle de usuários
dashboard de métricas da IA
comparação entre diferentes modelos de embeddings
avaliação de outros LLMs como fallback
deploy da aplicação
```

---

# Tecnologias

- Python
- FastAPI
- SQLite
- SQLAlchemy
- spaCy
- Sentence Transformers
- MiniLM
- NLP
- Embeddings
- Similaridade por cosseno
- Ollama
- Qwen
- HTML
- CSS
- JavaScript
- Git
- GitHub

---

# Contexto

Este projeto foi desenvolvido como estudo de **Inteligência Artificial aplicada à triagem de chamados técnicos**.

A proposta é demonstrar a evolução de uma solução partindo de um experimento simples com embeddings até uma arquitetura híbrida capaz de:

```text
entender o chamado
        ↓
extrair informações
        ↓
classificar semanticamente
        ↓
identificar ambiguidades
        ↓
recorrer a um LLM quando necessário
        ↓
rotear o chamado
        ↓
persistir os dados
        ↓
receber revisão humana
        ↓
medir e evoluir a qualidade do sistema
```

A separação em versões permite comparar as diferentes abordagens e entender quais componentes realmente contribuem para a solução.
