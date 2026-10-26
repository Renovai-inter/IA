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


def test_coleta_triagem_funcionario_equipe_repositories_get_snapshot():
    from datetime import datetime
    from decimal import Decimal
    from app.repository.postgresql.coleta_repository import ColetaRepository
    from app.repository.postgresql.triagem_repository import TriagemRepository
    from app.repository.postgresql.funcionario_repository import FuncionarioRepository
    from app.repository.postgresql.equipe_repository import EquipeRepository

    mock_db = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_db.connection.return_value.__enter__.return_value = mock_conn
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    coop_id = uuid4()
    func_id = uuid4()
    eq_id = uuid4()
    coleta_id = uuid4()
    triagem_id = uuid4()
    mat_id = uuid4()
    agora = datetime(2026, 10, 24, 12, 0)

    # 1. FuncionarioRepository
    mock_cursor.fetchall.return_value = [
        {
            "funcionario_id": func_id,
            "usuario_id": uuid4(),
            "cooperativa_id": coop_id,
            "nome": "João Cooperado",
            "email": "joao@coop.com",
            "cpf": "123.456.789-00",
            "cargo_id": uuid4(),
            "cargo": "Cooperado",
            "status_funcionario": "ATIVO",
            "esta_ativo": True,
            "esta_afastado": False,
            "data_admissao": agora,
            "data_desligamento": None,
        }
    ]
    snap_func = FuncionarioRepository(mock_db).get_snapshot(coop_id)
    assert len(snap_func.itens) == 1
    assert snap_func.itens[0].nome == "João Cooperado"

    # 2. EquipeRepository
    mock_cursor.fetchall.return_value = [
        {
            "equipe_id": eq_id,
            "cooperativa_id": coop_id,
            "gestor_id": func_id,
            "nome_gestor": "João Cooperado",
            "nome": "Equipe Alfa",
            "descricao": "Triagem",
            "data_criacao": agora,
            "esta_ativa": True,
            "cooperado_ids": [func_id],
        }
    ]
    snap_eq = EquipeRepository(mock_db).get_snapshot(coop_id)
    assert len(snap_eq.itens) == 1
    assert snap_eq.itens[0].cooperado_ids == [func_id]

    # 3. ColetaRepository
    mock_cursor.fetchall.return_value = [
        {
            "coleta_id": coleta_id,
            "cooperativa_id": coop_id,
            "cooperado_id": func_id,
            "cooperado_responsavel": "João Cooperado",
            "rota_id": None,
            "data_coleta": agora,
            "status_id": None,
            "status_atual": "CONCLUIDA",
            "tipo_coleta": "EXTERNA",
            "quantidade_kg": Decimal("450.000"),
            "imagem_url": None,
        }
    ]
    snap_col = ColetaRepository(mock_db).get_snapshot(coop_id)
    assert len(snap_col.itens) == 1
    assert snap_col.itens[0].coleta_id == coleta_id

    # 4. TriagemRepository
    mock_cursor.fetchall.return_value = [
        {
            "triagem_id": triagem_id,
            "cooperativa_id": coop_id,
            "equipe_id": eq_id,
            "nome_equipe": "Equipe Alfa",
            "coleta_id": coleta_id,
            "material_id": mat_id,
            "nome_categoria": "PET",
            "categoria_pai_id": None,
            "nome_categoria_pai": "Plástico",
            "data_triagem": agora,
            "status_id": None,
            "status_atual": "CONCLUIDA",
            "quantidade_kg": Decimal("430.000"),
            "quantidade_rejeito_kg": Decimal("20.000"),
            "imagem_url": None,
        }
    ]
    snap_tri = TriagemRepository(mock_db).get_snapshot(coop_id)
    assert len(snap_tri.itens) == 1
    assert snap_tri.itens[0].coleta_id == coleta_id
    assert snap_tri.itens[0].nome_categoria == "PET"
