# Testes — `renovai-ia`

## Estrutura

- `tests/unit/` — sem LLM, sem Ollama, sem rede, sem banco. Roteamento
  (`GraphBuilder`), parsing do protocolo `ROUTE=...` do `RouterAgent`,
  montagem de `EspecialistaOutput`. Usa dublês (`tests/support.py`) no
  lugar de `create_agent`/LLM real — roda em milissegundos, sempre.
- `tests/integration/` — roda contra um **Ollama local de verdade**
  (`app/llms/ollama_provider.py`). Repositórios (Postgres) continuam
  falsos (`tests/integration/fakes.py`) — só o LLM é real. Serve pra testar
  prompts e o comportamento ponta a ponta do grafo sem gastar quota do
  Gemini/Groq nem depender de rede externa.
- `tests/e2e/` — a FastAPI de produção de verdade, rodando contra os
  bancos **reais** do seu `.env` (Postgres/Mongo/Qdrant) + Ollama local.
  Cria e apaga seus próprios dados a cada teste. Só dispara com
  `E2E_CONFIRM=1` (trava de segurança, ver `tests/e2e/README.md`) — sem
  isso, é pulado, não falha.

## Rodando

```bash
uv sync
uv run pytest tests/unit
```

Para `tests/integration/`, além do `uv sync`:

```bash
ollama serve                            # se ainda não estiver rodando
ollama pull qwen2.5:3b-instruct         # modelo leve recomendado para testes locais rápidos
uv run pytest tests/integration
```

O `OllamaProvider` agora detecta automaticamente os modelos instalados no seu Ollama e possui timeout de segurança (default 120s). Se você quiser usar modelos específicos por tier ou tiver outros modelos baixados (ex: `llama3.2`, `qwen2.5:7b-instruct`), pode customizar via variáveis de ambiente:

```bash
# Usar o mesmo modelo leve em todos os tiers:
OLLAMA_MODEL="qwen2.5:3b-instruct" uv run pytest tests/integration

# Ou especificar por tier:
OLLAMA_MODEL_HIGH="qwen2.5:7b-instruct" OLLAMA_MODEL_LOW="qwen2.5:3b-instruct" uv run pytest tests/integration
```

Se o Ollama não estiver no ar em `http://localhost:11434` ou se o modelo configurado não estiver baixado, os testes de integração são **pulados** automaticamente com uma mensagem explicativa (não travam nem falham silenciosamente). Pra apontar para outro host/porta, defina `OLLAMA_BASE_URL`. Para alterar o tempo limite de inferência, defina `OLLAMA_TIMEOUT` (em segundos).

Para `tests/e2e/` (bancos reais do `.env` + Ollama) — leia
`tests/e2e/README.md` antes de rodar, ele explica a trava de segurança:

```bash
E2E_CONFIRM=1 uv run pytest tests/e2e -v
```

## O que NÃO está coberto ainda

- O agente Roteador tem um bug ativo conhecido (vaza o protocolo
  `ROUTE=...` no histórico de mensagens quando encaminha para um
  especialista) — documentado como teste `xfail` em
  `tests/unit/test_router_agent.py::test_router_agent_nao_deveria_vazar_protocolo_quando_encaminha`.
  Esse teste vira verde sozinho quando a correção for aplicada.
- Guardrail, Juiz, RAG e os demais agentes do roadmap não têm teste ainda
  porque não existem no código (ver a skill `ia-renovai` /
  `estado_atual_codigo.md` / `roadmap.md`).
- `tests/e2e/ollama_container.py` é uma cópia de `app/core/container.py`
  com o provider trocado, não um import com override — se
  `build_container` mudar (novo agente, novo repository), esta cópia pode
  ficar desatualizada. A alternativa seria adicionar um parâmetro
  `providers=` opcional em `build_container`; não fiz isso pra não mexer
  em código de produção sem você pedir explicitamente.
