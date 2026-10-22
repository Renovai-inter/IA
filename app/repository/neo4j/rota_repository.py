# repository/neo4j/rota_repository.py
import logging
from typing import Any, List, Optional
from uuid import uuid4

from neo4j import Driver

from app.ml.route_model import PontoParada, ResultadoRotaOtimizada
from app.ml.sp_zones import extrair_features_trecho, inferir_coordenadas_sp
from app.repository.base import Repository

log = logging.getLogger(__name__)


QUERY_SALVAR_ROTA = """
    MERGE (rt:Rota {id: $rota_id})
    SET rt.nome = $nome_rota,
        rt.coop_id = $cooperativa_id,
        rt.esta_ativa = true,
        rt.qtd_paradas = $qtd_paradas,
        rt.distancia_total_km = $distancia_total_km,
        rt.tempo_total_estimado_min = $tempo_total_estimado_min,
        rt.tempo_ordem_original_min = $tempo_ordem_original_min,
        rt.economia_estimada_min = $economia_estimada_min,
        rt.economia_percentual = $economia_percentual,
        rt.hora_saida = $hora_saida,
        rt.gerenciado_ia = true
    WITH rt
    OPTIONAL MATCH (rt)-[antiga:PARADA]->()
    DELETE antiga
    WITH rt
    FOREACH (_ IN CASE WHEN $cooperativa_id <> '' THEN [1] ELSE [] END |
        MERGE (c:Cooperativa {id: $cooperativa_id})
        MERGE (c)-[pr:POSSUI_ROTA]->(rt)
        SET pr.gerenciado_ia = true
    )
    WITH rt
    UNWIND $paradas AS p
    MERGE (pt:PontoDeColeta {id: p.ponto_id})
    SET pt:Local,
        pt.nome = coalesce(pt.nome, p.endereco),
        pt.logradouro = p.endereco,
        pt.bairro = p.bairro,
        pt.cep = p.cep,
        pt.lat = p.latitude,
        pt.lng = p.longitude,
        pt.location = point({latitude: p.latitude, longitude: p.longitude}),
        pt.gerenciado_ia = coalesce(pt.gerenciado_ia, true)
    MERGE (rt)-[rel:PARADA {ordem: p.ordem}]->(pt)
    SET rel.ponto_id = p.ponto_id,
        rel.nome_local = p.endereco,
        rel.zona_origem = p.zona_origem,
        rel.zona_destino = p.zona_destino,
        rel.distancia_haversine_km = p.distancia_haversine_km,
        rel.distancia_trecho_km = p.distancia_trecho_km,
        rel.distancia_acumulada_km = p.distancia_acumulada_km,
        rel.codigo_zona_origem = p.codigo_zona_origem,
        rel.codigo_zona_destino = p.codigo_zona_destino,
        rel.mudanca_zona = p.mudanca_zona,
        rel.fator_retencao = p.fator_retencao,
        rel.fator_horario_sp = p.fator_horario_sp,
        rel.hora_saida = p.hora_saida,
        rel.peso_estimado_kg = p.peso_estimado_kg,
        rel.tempo_trecho_min = p.tempo_trecho_min,
        rel.tempo_acumulado_min = p.tempo_acumulado_min,
        rel.tempo_real_min = coalesce(p.tempo_real_min, rel.tempo_real_min),
        rel.gerenciado_ia = true
    RETURN rt.id AS rota_id, count(rel) AS paradas_salvas
"""

QUERY_REGISTRAR_TEMPOS_REAIS = """
    UNWIND $execucoes AS item
    MATCH (rt:Rota {id: $rota_id})-[rel:PARADA]->(pt:PontoDeColeta)
    WHERE pt.id = item.ponto_id OR toString(rel.ordem) = toString(item.ponto_id)
    SET rel.tempo_real_min = toFloat(item.tempo_real_min),
        rel.gerenciado_ia = true
    RETURN count(rel) AS atualizados
"""

QUERY_BUSCAR_TRECHOS_TREINAMENTO = """
    MATCH (:Rota)-[rel:PARADA]->(:PontoDeColeta)
    WHERE rel.distancia_haversine_km IS NOT NULL
      AND rel.distancia_trecho_km IS NOT NULL
      AND rel.codigo_zona_origem IS NOT NULL
      AND rel.codigo_zona_destino IS NOT NULL
      AND rel.mudanca_zona IS NOT NULL
      AND rel.fator_retencao IS NOT NULL
      AND rel.fator_horario_sp IS NOT NULL
      AND rel.hora_saida IS NOT NULL
      AND rel.peso_estimado_kg IS NOT NULL
      AND (
          ($apenas_reais = true AND rel.tempo_real_min IS NOT NULL)
          OR ($apenas_reais = false AND coalesce(rel.tempo_real_min, rel.tempo_trecho_min) IS NOT NULL)
      )
    RETURN rel.distancia_haversine_km AS distancia_haversine_km,
           rel.distancia_trecho_km AS distancia_viaria_km,
           rel.codigo_zona_origem AS codigo_zona_origem,
           rel.codigo_zona_destino AS codigo_zona_destino,
           rel.mudanca_zona AS mudanca_zona,
           rel.fator_retencao AS fator_retencao,
           rel.fator_horario_sp AS fator_horario_sp,
           rel.hora_saida AS hora_saida,
           rel.peso_estimado_kg AS peso_estimado_kg,
           coalesce(rel.tempo_real_min, rel.tempo_trecho_min) AS tempo_total_min
"""


class RotaRepository(Repository[ResultadoRotaOtimizada, Driver]):
    def __init__(self, db: Driver, database: str = 'neo4j'):
        super().__init__(db)
        self._database = database

    def salvar_rota_otimizada(
        self,
        resultado: ResultadoRotaOtimizada,
        origem: PontoParada,
        paradas_entrada: List[PontoParada],
        cooperativa_id: str = '',
        rota_id: Optional[str] = None,
        nome_rota: str = 'Rota Otimizada IA',
    ) -> dict[str, Any]:
        """Persiste o nó :Rota e as arestas :PARADA no Neo4j com as features de ML do trecho."""
        log.debug('chegou no rota_repository para salvar rota otimizada no Neo4j')

        rota_id_final = rota_id or f'rota-{uuid4().hex[:10]}'
        mapa_entrada = {
            (p.id or p.endereco): p
            for p in paradas_entrada
        }

        lat_ant, lon_ant, zona_ant = inferir_coordenadas_sp(
            endereco=origem.endereco,
            bairro=origem.bairro,
            cep=origem.cep,
            latitude=origem.latitude,
            longitude=origem.longitude,
        )

        dist_acum = 0.0
        paradas_payload: list[dict[str, Any]] = []

        for p_otim in resultado.paradas_ordenadas:
            p_orig = mapa_entrada.get(p_otim.id) or mapa_entrada.get(p_otim.endereco)
            peso_kg = p_orig.peso_estimado_kg if p_orig else 200.0
            bairro = p_orig.bairro if p_orig else None
            cep = p_orig.cep if p_orig else None

            feats = extrair_features_trecho(
                lat_origem=lat_ant,
                lon_origem=lon_ant,
                zona_origem=zona_ant,
                lat_destino=p_otim.latitude,
                lon_destino=p_otim.longitude,
                zona_destino=p_otim.zona,
                hora_saida=resultado.hora_saida,
                peso_estimado_kg=peso_kg,
            )

            dist_acum = round(dist_acum + p_otim.distancia_trecho_km, 2)

            paradas_payload.append({
                'ordem':                  p_otim.ordem,
                'ponto_id':               p_otim.id,
                'endereco':               p_otim.endereco,
                'bairro':                 bairro,
                'cep':                    cep,
                'latitude':               p_otim.latitude,
                'longitude':              p_otim.longitude,
                'zona_origem':            zona_ant,
                'zona_destino':           p_otim.zona,
                'distancia_haversine_km': feats[0],
                'distancia_trecho_km':    p_otim.distancia_trecho_km,
                'distancia_acumulada_km': dist_acum,
                'codigo_zona_origem':     feats[2],
                'codigo_zona_destino':    feats[3],
                'mudanca_zona':           feats[4],
                'fator_retencao':         feats[5],
                'fator_horario_sp':       feats[6],
                'hora_saida':             feats[7],
                'peso_estimado_kg':       feats[8],
                'tempo_trecho_min':       p_otim.tempo_trecho_min,
                'tempo_acumulado_min':    p_otim.tempo_acumulado_min,
                'tempo_real_min':         None,
            })

            lat_ant, lon_ant, zona_ant = p_otim.latitude, p_otim.longitude, p_otim.zona

        with self._db.session(database=self._database) as session:
            registro = session.run(
                QUERY_SALVAR_ROTA,
                rota_id=rota_id_final,
                nome_rota=nome_rota,
                cooperativa_id=cooperativa_id or '',
                qtd_paradas=len(paradas_payload),
                distancia_total_km=resultado.distancia_total_km,
                tempo_total_estimado_min=resultado.tempo_total_estimado_min,
                tempo_ordem_original_min=resultado.tempo_ordem_original_min,
                economia_estimada_min=resultado.economia_estimada_min,
                economia_percentual=resultado.economia_percentual,
                hora_saida=resultado.hora_saida,
                paradas=paradas_payload,
            ).single()

        return {
            'rota_id':        registro['rota_id'] if registro else rota_id_final,
            'paradas_salvas': int(registro['paradas_salvas']) if registro else len(paradas_payload),
        }

    def registrar_tempos_reais(
        self,
        rota_id: str,
        tempos_reais_min: dict[str, float],
    ) -> int:
        """Atualiza o tempo real observado (em minutos) nas arestas :PARADA de uma rota no Neo4j."""
        if not tempos_reais_min:
            return 0

        execucoes = [
            {'ponto_id': str(k), 'tempo_real_min': float(v)}
            for k, v in tempos_reais_min.items()
        ]
        with self._db.session(database=self._database) as session:
            registro = session.run(
                QUERY_REGISTRAR_TEMPOS_REAIS,
                rota_id=rota_id,
                execucoes=execucoes,
            ).single()
        return int(registro['atualizados']) if registro else 0

    def buscar_trechos_treinamento(
        self,
        apenas_reais: bool = False,
    ) -> tuple[list[list[float]], list[float]]:
        """Extrai do grafo Neo4j os vetores de features (X) e tempos (y) das arestas :PARADA
        para retroalimentar o treinamento supervisionado do modelo de rotas."""
        X: list[list[float]] = []
        y: list[float] = []

        with self._db.session(database=self._database) as session:
            resultado = session.run(
                QUERY_BUSCAR_TRECHOS_TREINAMENTO,
                apenas_reais=apenas_reais,
            )
            for row in resultado:
                X.append([
                    float(row['distancia_haversine_km']),
                    float(row['distancia_viaria_km']),
                    float(row['codigo_zona_origem']),
                    float(row['codigo_zona_destino']),
                    float(row['mudanca_zona']),
                    float(row['fator_retencao']),
                    float(row['fator_horario_sp']),
                    float(row['hora_saida']),
                    float(row['peso_estimado_kg']),
                ])
                y.append(float(row['tempo_total_min']))

        return X, y
