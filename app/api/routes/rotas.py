import asyncio
from fastapi import APIRouter, Request

from app.api.schemas.rota_request import OtimizarRotaRequest, RetreinarRotaRequest
from app.api.schemas.rota_response import OtimizarRotaResponse, RetreinarRotaResponse

router = APIRouter(prefix='/rotas', tags=['rotas'])


@router.post('/otimizar', response_model=OtimizarRotaResponse)
async def otimizar_rota(request: Request, body: OtimizarRotaRequest) -> OtimizarRotaResponse:
    container = request.app.state.container
    modelo_rotas = container.modelo_rotas
    rota_repository = container.repositories['rota_repository']

    resultado = await asyncio.to_thread(
        modelo_rotas.otimizar_rota,
        body.origem,
        body.paradas,
        body.hora_saida,
    )

    rota_id = body.rota_id
    salvo_no_neo4j = False
    paradas_salvas = 0

    if body.salvar_no_neo4j:
        info_persistencia = await asyncio.to_thread(
            rota_repository.salvar_rota_otimizada,
            resultado,
            body.origem,
            body.paradas,
            body.cooperativa_id,
            body.rota_id,
            body.nome_rota,
        )
        rota_id = info_persistencia['rota_id']
        paradas_salvas = info_persistencia['paradas_salvas']
        salvo_no_neo4j = True

    return OtimizarRotaResponse(
        rota_id=rota_id,
        origem=resultado.origem,
        hora_saida=resultado.hora_saida,
        paradas_ordenadas=resultado.paradas_ordenadas,
        distancia_total_km=resultado.distancia_total_km,
        tempo_total_estimado_min=resultado.tempo_total_estimado_min,
        tempo_ordem_original_min=resultado.tempo_ordem_original_min,
        economia_estimada_min=resultado.economia_estimada_min,
        economia_percentual=resultado.economia_percentual,
        salvo_no_neo4j=salvo_no_neo4j,
        paradas_salvas_neo4j=paradas_salvas,
    )


@router.post('/retreinar', response_model=RetreinarRotaResponse)
async def retreinar_modelo_rotas(request: Request, body: RetreinarRotaRequest) -> RetreinarRotaResponse:
    container = request.app.state.container
    modelo_rotas = container.modelo_rotas
    rota_repository = container.repositories['rota_repository']

    atualizados = 0
    if body.rota_id and body.tempos_reais_min:
        atualizados = await asyncio.to_thread(
            rota_repository.registrar_tempos_reais,
            body.rota_id,
            body.tempos_reais_min,
        )

    metricas = await asyncio.to_thread(
        modelo_rotas.treinar_com_neo4j,
        rota_repository,
        body.incluir_csv_base,
        body.apenas_reais,
        True,
    )

    return RetreinarRotaResponse(
        status='ok',
        tempos_reais_atualizados=atualizados,
        mae_min=float(metricas['mae_min']),
        r2_score=float(metricas['r2_score']),
        amostras_neo4j=int(metricas['amostras_neo4j']),
        amostras_csv=int(metricas['amostras_csv']),
        amostras_treino=int(metricas['amostras_treino']),
        amostras_teste=int(metricas['amostras_teste']),
    )
