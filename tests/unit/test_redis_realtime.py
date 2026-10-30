"""Testes unitários para Rate Limiting, Fila de Processamento e Ranking em Tempo Real (Redis)."""

from __future__ import annotations

from app.repository.redis.realtime_repository import RedisRealtimeRepository


def test_rate_limiting_sliding_window():
    repo = RedisRealtimeRepository(
        db=None,
        max_requests=3,
        window_seconds=60,
    )

    r1 = repo.verificar_rate_limit("user_teste")
    r2 = repo.verificar_rate_limit("user_teste")
    r3 = repo.verificar_rate_limit("user_teste")
    r4 = repo.verificar_rate_limit("user_teste")

    assert r1.permitido is True
    assert r1.restante == 2
    assert r2.permitido is True
    assert r2.restante == 1
    assert r3.permitido is True
    assert r3.restante == 0

    assert r4.permitido is False
    assert r4.restante == 0
    assert r4.retry_after_segundos >= 1


def test_fila_e_ranking_tempo_real():
    repo = RedisRealtimeRepository(
        db=None,
        max_requests=10,
        window_seconds=60,
    )

    repo.enfileirar_turno(
        turno_id="t12345678",
        session_id="sess_1",
        user_id="usr_1",
        entidade_id="1",
        tipo_fluxo="COOPERATIVA",
        pergunta="Qual o saldo financeiro?",
    )

    painel_ativo = repo.obter_painel_realtime()
    assert painel_ativo["fila_processamento"]["tamanho_fila_ativa"] == 1

    repo.finalizar_turno_fila(
        turno_id="t12345678",
        session_id="sess_1",
        rota="financeiro",
        status="ok",
        latencia_ms=420,
        user_id="usr_1",
        entidade_id="1",
        tipo_fluxo="COOPERATIVA",
    )
    repo.registrar_interacao_ranking(
        entidade_id="1",
        tipo_fluxo="COOPERATIVA",
        rotas=["financeiro"],
        user_id="usr_1",
    )

    painel_final = repo.obter_painel_realtime()
    assert painel_final["fila_processamento"]["tamanho_fila_ativa"] == 0
    assert len(painel_final["fila_processamento"]["ultimos_processados"]) == 1
    assert painel_final["rankings_tempo_real"]["entidades_mais_ativas"][0]["membro"] == "COOPERATIVA:1"
    assert painel_final["rankings_tempo_real"]["rotas_mais_demandadas"][0]["membro"] == "financeiro"
