# tests/e2e — ponta a ponta, banco real + Ollama local

## O que é diferente de tests/integration

`tests/integration/` usa repositórios falsos (não toca em banco nenhum).
`tests/e2e/` sobe a **FastAPI de produção de verdade** (`chat.router` +
`sessions.router`, exatamente os mesmos handlers de `app/main.py`) e fala
com os bancos **reais** configurados no seu `.env` — `DATABASE_URL`,
`MONGODB_URI`, `QDRANT_URL`. A única coisa trocada é o LLM: em vez de
Gemini/Groq, usa Ollama local (`tests/e2e/ollama_container.py`), pra não
gastar quota de API a cada rodada.

**Isso significa que esta suíte escreve no banco que estiver apontado no
seu `.env`.** Se isso for produção, é produção mesmo — foi uma escolha
deliberada (pra não complicar com docker-compose/infra isolada), não um
descuido. Os dados que ela grava são sempre criados e apagados pelo
próprio teste (`tests/e2e/seed.py`), com nomes marcados
(`[E2E PYTEST ...]`) e limpeza garantida via `try/finally` mesmo se o
teste falhar no meio — mas ainda assim é escrita real. Se algum dia você
tiver um banco de homologação separado, é só apontar `DATABASE_URL` (e os
outros) pra ele antes de rodar isto.

## Trava de segurança

Por isso a suíte inteira (`tests/e2e/conftest.py::pytestmark`) só roda se
a env var `E2E_CONFIRM=1` estiver definida — sem ela, todo teste aqui é
**pulado**, não falha. Isso evita que `pytest` na raiz do projeto dispare
escrita em produção sem querer; `tests/unit` e `tests/integration`
continuam rodando normalmente nesse caso.

## Rodando

```bash
uv sync
ollama serve                       # se ainda não estiver rodando
ollama pull qwen2.5:14b-instruct   # tier HIGH
ollama pull qwen2.5:7b-instruct    # tier MEDIUM
ollama pull qwen2.5:3b-instruct    # tier LOW

E2E_CONFIRM=1 uv run pytest tests/e2e -v
```

No Windows (PowerShell):

```powershell
$env:E2E_CONFIRM = "1"
uv run pytest tests/e2e -v
```

## Se algo falhar no meio (dados órfãos)

`cenario_estoque` (tests/e2e/conftest.py) sempre chama `limpar_cenario`
num `finally`, então mesmo um teste que falha limpa depois de si. Só fica
órfão se o processo for interrompido à força (Ctrl+C bem no meio, queda de
conexão com o Postgres). Se isso acontecer, procure por linhas marcadas
`[E2E PYTEST ...]` em `cooperativas` e `categorias_materiais` — são as
únicas tabelas onde o nome fica gravado — e apague manualmente (na ordem:
`estoques`/`movimentacoes_estoques` → `materiais` → `categorias_materiais`
→ `perfis` → `cooperativas`, por causa das FKs).

## Limitação conhecida (herdada, não introduzida aqui)

`ResumoService` (`app/memory/resumo_service.py`) usa o mesmo LLM de chat
(tier LOW) tanto pra gerar resumo em texto quanto pra gerar embeddings
(`.embed_query`) — isso já não funcionava com `ChatGroq` em produção, e
também não funciona com `ChatOllama` aqui. Só é exercitado se um agente
decidir chamar a tool `buscar_historico` com um termo de busca (memória
via Qdrant); o fluxo comum de `POST /chat` testado aqui não passa por
isso. Não é algo que estes testes e2e tentam corrigir.
