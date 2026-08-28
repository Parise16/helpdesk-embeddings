# HelpDesk Embeddings + NLP V2

Versão experimental de um classificador de chamados técnicos que combina:

- **spaCy** para NLP estruturado;
- **MiniLM multilíngue** para embeddings;
- **similaridade semântica** entre chamados;
- sinais lexicais/entidades como evidência auxiliar;
- **Qwen 4B via Ollama** apenas como fallback em casos ambíguos.

## Ideia principal

A V2 não substitui a frase original por palavras lematizadas antes do embedding.

Isso é proposital: modelos Sentence Transformer usam o contexto da frase. Remover
palavras indiscriminadamente pode eliminar informação útil.

O pipeline usa duas fontes de evidência:

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

## O que o NLP extrai

Exemplo:

```text
Meu notebook não conecta no Wi-Fi do laboratório 704 desde ontem.
```

Saída estruturada aproximada:

```json
{
  "texto_normalizado": "notebook não conectar wi-fi laboratório 704 ontem",
  "locais": ["laboratório 704"],
  "tecnologias": ["Wi-Fi"],
  "dispositivos": ["notebook"],
  "tempo": ["desde ontem"]
}
```

## Categorias atuais

- Rede e Internet
- VPN
- Hardware
- Acesso e Autenticação
- Software e Aplicações
- Banco de Dados
- Segurança

## Instalação

```powershell
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
python -m spacy download pt_core_news_sm
```

O Sentence Transformer será baixado automaticamente na primeira execução se ainda
não estiver no cache.

## Ollama

O fallback padrão é:

```text
qwen3:4b-instruct
```

Confira:

```powershell
ollama list
```

## Execução

```powershell
python main.py
```

Para ver a estrutura NLP completa:

```powershell
python main.py --debug-nlp
```

## Exemplo de teste

```text
Meu notebook não conecta no Wi-Fi do laboratório 704 desde ontem.
```

O programa mostra:

- texto normalizado;
- local detectado;
- tecnologias/dispositivos;
- Top 3 categorias;
- score do embedding;
- sinal NLP;
- score híbrido;
- problema extraído;
- prioridade;
- se o Qwen precisou ser acionado.

## Por que o NLP não substitui o embedding?

O spaCy ajuda a estruturar informações como local, tecnologia, dispositivo e termos
relevantes. Já o MiniLM é responsável por entender a semântica da frase completa.

A V2 combina os dois sinais em vez de destruir o contexto antes de gerar o embedding.

## Próximo passo

Esta versão é propositalmente independente de SQL. Depois de avaliar a qualidade,
o pipeline pode ser incorporado ao AI HelpDesk Triage com banco de dados e FastAPI.
