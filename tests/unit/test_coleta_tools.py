"""Testes unitários das tools de Coleta, Triagem, Equipes e Cooperados."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.repository.postgresql.coleta_repository import ColetaSnapshot, Coleta
from app.repository.postgresql.triagem_repository import TriagemSnapshot, Triagem
from app.repository.postgresql.funcionario_repository import FuncionarioSnapshot, Funcionario
from app.repository.postgresql.equipe_repository import EquipeSnapshot, Equipe
from app.repository.postgresql.material_repository import MaterialSnapshot, Material
from app.tools.coleta_tools import ColetaToolkit


@pytest.fixture
def snapshots_operacionais():
    coop_id = uuid4()
    gestor_id = uuid4()
    coop1_id = uuid4()
    coop2_id = uuid4()
    cargo_coop_id = uuid4()
    cargo_gestor_id = uuid4()
    equipe1_id = uuid4()
    equipe2_id = uuid4()
    mat_pet_id = uuid4()
    mat_pap_id = uuid4()
    cat_plastico_id = uuid4()
    cat_pet_id = uuid4()
    cat_papelao_id = uuid4()

    f_gestor = Funcionario(
        funcionario_id=gestor_id,
        usuario_id=uuid4(),
        cooperativa_id=coop_id,
        nome='Carlos Gestor',
        email='carlos@coop.com',
        cpf='111.111.111-11',
        cargo_id=cargo_gestor_id,
        cargo='Gestor',
        status_funcionario='ATIVO',
        esta_ativo=True,
        esta_afastado=False,
        data_admissao=datetime(2025, 1, 10, 8, 0),
    )
    f_coop1 = Funcionario(
        funcionario_id=coop1_id,
        usuario_id=uuid4(),
        cooperativa_id=coop_id,
        nome='João da Silva',
        email='joao@coop.com',
        cpf='222.222.222-22',
        cargo_id=cargo_coop_id,
        cargo='Cooperado',
        status_funcionario='ATIVO',
        esta_ativo=True,
        esta_afastado=False,
        data_admissao=datetime(2025, 3, 15, 8, 0),
    )
    f_coop2 = Funcionario(
        funcionario_id=coop2_id,
        usuario_id=uuid4(),
        cooperativa_id=coop_id,
        nome='Maria Oliveira',
        email='maria@coop.com',
        cpf='333.333.333-33',
        cargo_id=cargo_coop_id,
        cargo='Cooperado',
        status_funcionario='ATIVO',
        esta_ativo=True,
        esta_afastado=False,
        data_admissao=datetime(2025, 4, 1, 8, 0),
    )

    eq1 = Equipe(
        equipe_id=equipe1_id,
        cooperativa_id=coop_id,
        gestor_id=gestor_id,
        nome_gestor='Carlos Gestor',
        nome='Equipe Alfa',
        descricao='Equipe de triagem de plásticos e papéis',
        data_criacao=datetime(2025, 2, 1, 9, 0),
        esta_ativa=True,
        cooperado_ids=[coop1_id],
    )
    eq2 = Equipe(
        equipe_id=equipe2_id,
        cooperativa_id=coop_id,
        gestor_id=gestor_id,
        nome_gestor='Carlos Gestor',
        nome='Equipe Beta',
        descricao='Equipe de coleta externa',
        data_criacao=datetime(2025, 2, 1, 9, 0),
        esta_ativa=True,
        cooperado_ids=[coop2_id],
    )

    m_pet = Material(
        material_id=mat_pet_id,
        categoria_id=cat_pet_id,
        nome_categoria='PET',
        categoria_pai_id=cat_plastico_id,
        nome_categoria_pai='Plástico',
        cooperativa_id=coop_id,
        nome_cooperativa='Coop Teste',
        preco_sugerido=Decimal('3.50'),
        esta_disponivel=True,
    )
    m_pap = Material(
        material_id=mat_pap_id,
        categoria_id=cat_papelao_id,
        nome_categoria='Papelão',
        categoria_pai_id=None,
        nome_categoria_pai=None,
        cooperativa_id=coop_id,
        nome_cooperativa='Coop Teste',
        preco_sugerido=Decimal('1.20'),
        esta_disponivel=True,
    )

    c1 = Coleta(
        coleta_id=uuid4(),
        cooperativa_id=coop_id,
        cooperado_id=coop1_id,
        cooperado_responsavel='João da Silva',
        rota_id=uuid4(),
        data_coleta=datetime(2026, 10, 24, 10, 0),
        status_atual='CONCLUIDA',
        tipo_coleta='EXTERNA',
        quantidade_kg=Decimal('520'),
    )
    c2 = Coleta(
        coleta_id=uuid4(),
        cooperativa_id=coop_id,
        cooperado_id=coop2_id,
        cooperado_responsavel='Maria Oliveira',
        rota_id=None,
        data_coleta=datetime(2026, 10, 24, 14, 0),
        status_atual='AGENDADA',
        tipo_coleta='ENTREGA',
        quantidade_kg=Decimal('300'),
    )

    # 1 coleta (c1) com 2 triagens (PET + Papelão)
    t1 = Triagem(
        triagem_id=uuid4(),
        cooperativa_id=coop_id,
        equipe_id=equipe1_id,
        nome_equipe='Equipe Alfa',
        coleta_id=c1.coleta_id,
        material_id=mat_pet_id,
        nome_categoria='PET',
        categoria_pai_id=cat_plastico_id,
        nome_categoria_pai='Plástico',
        data_triagem=datetime(2026, 10, 24, 11, 0),
        status_atual='CONCLUIDA',
        quantidade_kg=Decimal('300'),
        quantidade_rejeito_kg=Decimal('15'),
    )
    t2 = Triagem(
        triagem_id=uuid4(),
        cooperativa_id=coop_id,
        equipe_id=equipe1_id,
        nome_equipe='Equipe Alfa',
        coleta_id=c1.coleta_id,
        material_id=mat_pap_id,
        nome_categoria='Papelão',
        categoria_pai_id=None,
        nome_categoria_pai=None,
        data_triagem=datetime(2026, 10, 24, 11, 30),
        status_atual='CONCLUIDA',
        quantidade_kg=Decimal('200'),
        quantidade_rejeito_kg=Decimal('5'),
    )

    agora = datetime.now()
    return {
        'coleta_snapshot': ColetaSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[c1, c2]),
        'triagem_snapshot': TriagemSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[t1, t2]),
        'funcionario_snapshot': FuncionarioSnapshot(
            cooperativa_id=coop_id, capturado_em=agora, itens=[f_gestor, f_coop1, f_coop2]
        ),
        'equipe_snapshot': EquipeSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[eq1, eq2]),
        'material_snapshot': MaterialSnapshot(cooperativa_id=coop_id, capturado_em=agora, itens=[m_pet, m_pap]),
    }


def test_consultar_coletas_filtro_status_e_triagens_vinculadas(snapshots_operacionais):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': snapshots_operacionais}}

    res = toolkit.consultar_coletas(config=config, status='CONCLUIDA')
    assert res['status'] == 'ok'
    assert res['total_coletas_encontradas'] == 1
    assert res['peso_total_coletado_kg'] == 520.0
    coleta = res['coletas'][0]
    assert coleta['cooperado_responsavel'] == 'João da Silva'
    assert coleta['total_triagens_vinculadas'] == 2
    assert coleta['peso_triado_kg'] == 500.0
    assert coleta['peso_rejeito_kg'] == 20.0
    assert coleta['materiais_triados'] == ['PET', 'Papelão']


def test_consultar_triagens_por_material_e_taxa_rejeito(snapshots_operacionais):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': snapshots_operacionais}}

    res_todas = toolkit.consultar_triagens(config=config)
    assert res_todas['status'] == 'ok'
    assert res_todas['total_triagens_encontradas'] == 2
    assert res_todas['peso_total_triado_kg'] == 500.0
    assert res_todas['peso_total_rejeito_kg'] == 20.0
    assert res_todas['taxa_rejeito_percentual'] == round(20 / 520 * 100, 2)
    assert res_todas['quantidade_por_categoria_kg'] == {'PET': 300.0, 'Papelão': 200.0}

    # Filtro por macrocategoria ('plástico' -> resolve PET via material_snapshot)
    res_plastico = toolkit.consultar_triagens(config=config, material_nomes=['plástico'])
    assert res_plastico['status'] == 'ok'
    assert res_plastico['total_triagens_encontradas'] == 1
    assert res_plastico['triagens'][0]['nome_categoria'] == 'PET'


def test_consultar_equipes_cooperados(snapshots_operacionais):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': snapshots_operacionais}}

    res = toolkit.consultar_equipes_cooperados(config=config, equipe_nome='Alfa')
    assert res['status'] == 'ok'
    assert res['total_equipes'] == 1
    eq = res['equipes'][0]
    assert eq['nome'] == 'Equipe Alfa'
    assert eq['nome_gestor'] == 'Carlos Gestor'
    assert eq['numero_cooperados'] == 1
    assert eq['cooperados'][0]['nome'] == 'João da Silva'


def test_resumo_operacional(snapshots_operacionais):
    toolkit = ColetaToolkit()
    config = {'configurable': {'snapshots': snapshots_operacionais}}

    res = toolkit.resumo_operacional(config=config)
    assert res['status'] == 'ok'
    assert res['total_coletas'] == 2
    assert res['peso_coletado_kg'] == 820.0
    assert res['total_triagens'] == 2
    assert res['peso_triado_kg'] == 500.0
    assert res['peso_rejeito_kg'] == 20.0
    assert res['equipes_ativas'] == 2
    assert res['cooperados_ativos'] == 3
    assert res['taxa_aproveitamento_percentual'] > 90.0
