"""Conftest de topo — garante que o pacote `app` (e `tests`) sejam
importáveis independente de como o pytest for invocado (raiz do repo,
de dentro de tests/, via IDE, etc). `[tool.pytest.ini_options].pythonpath`
no pyproject.toml já cobre o caso comum; isto aqui é só reforço.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MOCK_ENV_VARS = {
    "GEMINI_API_KEY": "mock-gemini-key",
    "GROQ_API_KEY": "mock-groq-key",
    "QDRANT_API_KEY": "mock-qdrant-key",
    "DATABASE_URL": "postgresql://user:pass@localhost:5432/test_db",
    "QDRANT_URL": "http://localhost:6333",
    "MONGODB_URI": "mongodb://localhost:27017/test",
    "FAQ_PATH": "data/faq.txt",
}

for key, value in MOCK_ENV_VARS.items():
    os.environ.setdefault(key, value)