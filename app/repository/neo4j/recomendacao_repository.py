# repository/neo4j/recomendacao_repository.py
import logging
from typing import List, Optional
from neo4j import Driver
from pydantic import BaseModel

from app.repository.base import Repository

log = logging.getLogger(__name__)


QUERY_RECOMENDAR_COOPERATIVAS = """
    MATCH (c:Cooperativa)-[proc:PROCESSA_MATERIAL]->(m:Material)
    WHERE toLower(m.nome) CONTAINS toLower($material)
      AND (
        coalesce(proc.estoque_atual_kg, 0.0) >= $volume_minimo * 0.5
        OR coalesce(proc.capacidade_mensal_kg, 0.0) >= $volume_minimo
      )
    OPTIONAL MATCH (e:EmpresaCompradora {id: $empresa_id})
    OPTIONAL MATCH p = shortestPath((e)-[:CONECTADA_A*..6]-(c))
    OPTIONAL MATCH (e)-[fav:FAVORITA]->(c)
    OPTIONAL MATCH (c)-[col:COLETA_EM]->(:PontoColeta)
    WITH c, m, proc, fav,
         count(DISTINCT col) AS fontes_validadas,
         coalesce(
             reduce(dist = 0.0, r IN relationships(p) | dist + coalesce(r.distancia_km, 0.0)),
             c.distancia_padrao_km
         ) AS distancia_km
    WHERE distancia_km IS NOT NULL AND distancia_km <= $raio_max_km
    RETURN c.id AS cooperativa_id,
           c.nome AS cooperativa_nome,
           m.nome AS material_nome,
           coalesce(proc.capacidade_mensal_kg, 0.0) AS capacidade_mensal_kg,
           coalesce(proc.estoque_atual_kg, 0.0) AS estoque_atual_kg,
           proc.preco_sugerido_medio AS preco_sugerido_medio,
           distancia_km,
           coalesce(c.score_conformidade, 0.0) AS score_conformidade,
           fontes_validadas,
           (fav IS NOT NULL) AS favorita
    LIMIT $limit
"""


def _clamp01(valor: float) -> float:
    return max(0.0, min(1.0, float(valor)))


def calcular_score_multicriterio(
    *,
    estoque_atual_kg: float,
    volume_minimo_kg: float,
    distancia_km: float,
    raio_max_km: float,
    capacidade_mensal_kg: float,
    score_conformidade: float,
    fontes_validadas: int = 0,
    favorita: bool = False,
) -> float:
    """Calcula o score multicritério ponderado (0.0 a 1.0) alinhado à query Cypher do RPA:
      - 35% Disponibilidade imediata em estoque (estoque_atual_kg / volume_minimo_kg)
      - 30% Proximidade logística na malha viária (1 - distancia_km / raio_max_km)
      - 15% Capacidade mensal estrutural (capacidade_mensal_kg / (volume_minimo_kg * 2))
      - 10% Score de conformidade e histórico de entregas (0..1)
      -  5% Maturidade de captação (fontes_validadas / 5)
      -  5% Bônus de cooperativa favorita (parceria prévia)
    """
    vol_ref = max(float(volume_minimo_kg), 1.0)
    raio_ref = max(float(raio_max_km), 1.0)

    score_disp = _clamp01(float(estoque_atual_kg) / vol_ref)
    score_dist = _clamp01(1.0 - (float(distancia_km) / raio_ref))
    score_cap = _clamp01(float(capacidade_mensal_kg) / (vol_ref * 2.0))
    score_conf = _clamp01(float(score_conformidade))
    score_fontes = _clamp01(float(fontes_validadas) / 5.0)
    bonus_fav = 1.0 if favorita else 0.0

    score = (
        0.35 * score_disp
        + 0.30 * score_dist
        + 0.15 * score_cap
        + 0.10 * score_conf
        + 0.05 * score_fontes
        + 0.05 * bonus_fav
    )
    return round(score, 4)


class RecomendacaoCooperativa(BaseModel):
    cooperativa_id: str
    cooperativa_nome: str
    material: str
    capacidade_mensal_kg: float
    estoque_atual_kg: float
    preco_sugerido_medio: Optional[float] = None
    distancia_km: float
    score_conformidade: float
    fontes_validadas: int = 0
    favorita: bool = False
    score_final: float


class RecomendacaoRepository(Repository[RecomendacaoCooperativa, Driver]):
    def __init__(self, db: Driver, database: str = 'neo4j'):
        super().__init__(db)
        self._database = database

    def recomendar_cooperativas(
        self,
        material: str,
        empresa_id: Optional[str] = None,
        raio_max_km: float = 50.0,
        volume_minimo_kg: float = 500.0,
        limit: int = 5,
    ) -> List[RecomendacaoCooperativa]:
        log.debug('chegou no recomendacao_repository e consultou grafo Neo4j')

        with self._db.session(database=self._database) as session:
            result = session.run(
                QUERY_RECOMENDAR_COOPERATIVAS,
                material=material,
                empresa_id=empresa_id or '',
                volume_minimo=float(volume_minimo_kg),
                raio_max_km=float(raio_max_km),
                limit=max(int(limit) * 2, 10),
            )

            recomendacoes: List[RecomendacaoCooperativa] = []
            for record in result:
                dist = float(record['distancia_km'])
                conf = float(record['score_conformidade'] or 0.0)
                cap = float(record['capacidade_mensal_kg'] or 0.0)
                est = float(record['estoque_atual_kg'] or 0.0)
                preco = record.get('preco_sugerido_medio')
                fontes = int(record['fontes_validadas'] or 0)
                fav = bool(record['favorita'])

                score = calcular_score_multicriterio(
                    estoque_atual_kg=est,
                    volume_minimo_kg=volume_minimo_kg,
                    distancia_km=dist,
                    raio_max_km=raio_max_km,
                    capacidade_mensal_kg=cap,
                    score_conformidade=conf,
                    fontes_validadas=fontes,
                    favorita=fav,
                )

                recomendacoes.append(
                    RecomendacaoCooperativa(
                        cooperativa_id=str(record['cooperativa_id']),
                        cooperativa_nome=str(record['cooperativa_nome']),
                        material=str(record['material_nome']),
                        capacidade_mensal_kg=cap,
                        estoque_atual_kg=est,
                        preco_sugerido_medio=float(preco) if preco is not None else None,
                        distancia_km=round(dist, 2),
                        score_conformidade=round(conf, 2),
                        fontes_validadas=fontes,
                        favorita=fav,
                        score_final=score,
                    )
                )

        recomendacoes.sort(
            key=lambda r: (-r.score_final, r.distancia_km, -r.estoque_atual_kg)
        )
        return recomendacoes[:limit]
