"""Conftest de topo — garante que o pacote `app` (e `tests`) sejam
importáveis independente de como o pytest for invocado (raiz do repo,
de dentro de tests/, via IDE, etc). `[tool.pytest.ini_options].pythonpath`
no pyproject.toml já cobre o caso comum; isto aqui é só reforço.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))