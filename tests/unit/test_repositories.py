"""
Testes unitários dos repositórios PostgreSQL do Renovai.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4
import pytest

from app.repository.postgresql.material_repository import MaterialRepository, _QUERY_MATERIAL_POR_COOPERATIVA
from app.repository.postgresql.perfil_repository import PerfilRepository


def test_material_repository_query_inclui_globais():
    # Garante que a query sempre busca os materiais customizados da cooperativa E os globais (cooperativa_id IS NULL)
    assert "m.cooperativa_id = %s OR m.cooperativa_id IS NULL" in _QUERY_MATERIAL_POR_COOPERATIVA


def test_material_repository_get_snapshot():
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_db.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    coop_id = uuid4()
    mat_id1 = uuid4()
    mat_id2 = uuid4()
    cat_id = uuid4()

    mock_cursor.fetchall.return_value = [
        {
            "material_id": mat_id1,
            "categoria_id": cat_id,
            "nome_categoria": "Papelão",
            "categoria_pai_id": None,
            "nome_categoria_pai": None,
            "cooperativa_id": coop_id,
            "nome_cooperativa": "Coop Teste",
            "preco_sugerido": 1.70,
            "esta_disponivel": True,
            "imagem_url": None,
        },
        {
            "material_id": mat_id2,
            "categoria_id": cat_id,
            "nome_categoria": "Alumínio",
            "categoria_pai_id": None,
            "nome_categoria_pai": None,
            "cooperativa_id": None,
            "nome_cooperativa": None,
            "preco_sugerido": 5.50,
            "esta_disponivel": True,
            "imagem_url": None,
        },
    ]

    repo = MaterialRepository(mock_db)
    snapshot = repo.get_snapshot(coop_id)

    assert snapshot.cooperativa_id == coop_id
    assert len(snapshot.itens) == 2
    assert snapshot.itens[0].cooperativa_id == coop_id
    assert snapshot.itens[1].cooperativa_id is None


def test_perfil_repository_resolve_perfil_existente():
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_db.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    perfil_id = uuid4()
    coop_id = uuid4()

    mock_cursor.fetchone.return_value = {
        "empresa_id": None,
        "cooperativa_id": coop_id,
    }

    repo = PerfilRepository(mock_db)
    ctx = repo.resolve_perfil(perfil_id)

    assert ctx.perfil_id == perfil_id
    assert ctx.tipo == "COOPERATIVA"
    assert ctx.cooperativa_id == coop_id


def test_perfil_repository_resolve_perfil_inexistente_lanca_erro():
    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()

    mock_db.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    mock_cursor.fetchone.return_value = None

    repo = PerfilRepository(mock_db)
    with pytest.raises(ValueError, match="não encontrado"):
        repo.resolve_perfil(uuid4())
