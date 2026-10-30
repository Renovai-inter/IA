import asyncio
import json
import logging
import re
import unicodedata
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig

from app.agents.base import BaseAgent
from app.graph.state import EspecialistaOutput, GraphState, resolver_contexto_dominio
from app.ml.route_model import ModeloRotaSupervisionadoSP, PontoParada
from app.ml.sp_zones import BAIRROS_COORDS_SP
from app.repository.neo4j.rota_repository import RotaRepository

log = logging.getLogger('renovai.logistica')


def _normalizar_texto(texto: str) -> str:
    sem_acento = ''.join(
        ch for ch in unicodedata.normalize('NFKD', texto or '')
        if not unicodedata.combining(ch)
    )
    return re.sub(r'\s+', ' ', sem_acento).strip().lower()


_PADRAO_HORA = re.compile(
    r'(?:[aà]s\s+|saindo\s+[aà]s\s+|hor[aá]rio\s+)(\d{1,2})\s*(?:h(?:oras?)?|:00)?\b'
    r'|\b(\d{1,2})\s*h(?:oras?)?\b',
    re.IGNORECASE,
)

_PADRAO_CEP = re.compile(r'\b(0[1-58]\d{3}-?\d{3})\b')

_PADRAO_ORIGEM = re.compile(
    r'(?:saindo\s+d[aeo]s?\s+|partindo\s+d[aeo]s?\s+|origem\s+(?:em|n[ao])\s+)([a-zA-ZÀ-ÿ\s]{3,25})',
    re.IGNORECASE,
)


def _extrair_hora_saida(texto: str, padrao: int = 8) -> int:
    """Extrai o horário de saída (0..23) mencionado na mensagem, ou devolve o horário padrão."""
    for match in _PADRAO_HORA.finditer(texto or ''):
        valor = match.group(1) or match.group(2)
        if valor is not None:
            hora = int(valor)
            if 0 <= hora <= 23:
                return hora
    return padrao


def _extrair_locais_mensagem_sp(
    texto: str,
) -> tuple[Optional[PontoParada], List[PontoParada]]:
    """Extrai de forma determinística (sem LLM) a origem e as paradas em SP citadas na mensagem."""
    if not texto:
        return None, []

    texto_norm = _normalizar_texto(texto)

    bairro_origem_explicito: Optional[str] = None
    m_orig = _PADRAO_ORIGEM.search(texto)
    if m_orig:
        trecho_orig = _normalizar_texto(m_orig.group(1))
        for b_key in sorted(BAIRROS_COORDS_SP.keys(), key=len, reverse=True):
            if b_key in trecho_orig:
                bairro_origem_explicito = b_key
                break

    ocorrencias: list[tuple[int, PontoParada]] = []
    intervalos_ocupados: list[tuple[int, int]] = []

    for b_key in sorted(BAIRROS_COORDS_SP.keys(), key=len, reverse=True):
        padrao_b = re.compile(rf'\b{re.escape(b_key)}\b')
        for m in padrao_b.finditer(texto_norm):
            ini, fim = m.span()
            if any(not (fim <= a or ini >= b) for a, b in intervalos_ocupados):
                continue
            intervalos_ocupados.append((ini, fim))
            lat, lon, _ = BAIRROS_COORDS_SP[b_key]
            ocorrencias.append((
                ini,
                PontoParada(
                    id=f"bairro-{b_key.replace(' ', '-')}",
                    endereco=b_key.title(),
                    bairro=b_key.title(),
                    latitude=lat,
                    longitude=lon,
                    peso_estimado_kg=200.0,
                ),
            ))

    for m_cep in _PADRAO_CEP.finditer(texto):
        cep_str = m_cep.group(1)
        ocorrencias.append((
            m_cep.start(),
            PontoParada(
                id=f'cep-{cep_str}',
                endereco=f'CEP {cep_str}',
                cep=cep_str,
                peso_estimado_kg=200.0,
            ),
        ))

    ocorrencias.sort(key=lambda item: item[0])
    pontos = [p for _, p in ocorrencias]

    if not pontos:
        return None, []

    if bairro_origem_explicito:
        id_alvo = f"bairro-{bairro_origem_explicito.replace(' ', '-')}"
        origem = next((p for p in pontos if p.id == id_alvo), pontos[0])
        paradas = [p for p in pontos if p.id != origem.id]
        return origem, paradas

    return None, pontos


# Compatibilidade retroativa com testes que importam diretamente
extrair_hora_saida = _extrair_hora_saida
extrair_locais_mensagem_sp = _extrair_locais_mensagem_sp


class LogisticaAgent(BaseAgent):
    """Nó determinístico de logística que executa diretamente o pipeline
    supervisionado (StandardScaler + GradientBoostingRegressor + 2-Opt) sem invocar LLM."""

    nome_agente: str = 'logistica'
    descricao: str = 'Otimização supervisionada de rotas de coleta em São Paulo'

    def __init__(
        self,
        llm: Any = None,
        system_prompt: str = '',
        tools: list | None = None,
        modelo_rotas: Optional[ModeloRotaSupervisionadoSP] = None,
        rota_repository: Optional[RotaRepository] = None,
    ):
        super().__init__(llm=llm, system_prompt=system_prompt, tools=tools)
        self.modelo_rotas = modelo_rotas or ModeloRotaSupervisionadoSP()
        self.rota_repository = rota_repository

    def _obter_ultima_pergunta(self, state: GraphState) -> str:
        msgs = list(state.get('messages') or [])
        for msg in reversed(msgs):
            tipo = getattr(msg, 'type', None) or (msg.get('role') if isinstance(msg, dict) else None)
            if tipo in ('human', 'user'):
                return self._obter_texto_mensagem(msg) if hasattr(msg, 'content') else str(msg.get('content', ''))
        return self._obter_texto_mensagem(msgs[-1]) if msgs else ''

    def _executar_pipeline(self, state: GraphState, config: RunnableConfig | None = None) -> dict[str, Any]:
        pergunta = self._obter_ultima_pergunta(state)
        hora_saida = _extrair_hora_saida(pergunta, padrao=8)

        cooperativa_id, _, _ = resolver_contexto_dominio(state, config)

        origem_msg, paradas_msg = _extrair_locais_mensagem_sp(pergunta)

        dados_neo4j: Optional[dict[str, Any]] = None
        if cooperativa_id and self.rota_repository is not None:
            try:
                dados_neo4j = self.rota_repository.buscar_dados_rota_cooperativa(cooperativa_id)
            except Exception as exc:
                log.warning(
                    f'[logistica] Falha ao consultar rota prévia no Neo4j ({exc}). '
                    'Continuando com os pontos extraídos da mensagem.'
                )
                dados_neo4j = None

        origem: Optional[PontoParada] = origem_msg
        paradas: List[PontoParada] = list(paradas_msg)
        rota_id_neo4j: Optional[str] = None
        nome_rota_neo4j: str = 'Rota Otimizada IA'

        if dados_neo4j:
            rota_id_neo4j = dados_neo4j.get('rota_id')
            nome_rota_neo4j = dados_neo4j.get('nome_rota') or 'Rota Otimizada IA'
            if origem is None:
                origem = dados_neo4j.get('origem')
            if not paradas:
                paradas = list(dados_neo4j.get('paradas') or [])

        if origem is None and len(paradas) >= 2:
            origem = paradas[0]
            paradas = paradas[1:]

        if origem is None or len(paradas) == 0:
            payload_esclarecer = {
                'dominio': 'logistica',
                'intencao': 'otimizar_rota',
                'resposta': 'Não encontrei pontos de parada suficientes para calcular a rota otimizada.',
                'recomendacao': '',
                'esclarecer': (
                    'Informe os bairros ou CEPs de origem e de parada em São Paulo '
                    '(ex.: "Saindo do Tatuapé às 8h passando pela Mooca, Pinheiros e Sé").'
                ),
            }
            return self._empacotar_saida(payload_esclarecer)

        resultado = self.modelo_rotas.otimizar_rota(
            origem=origem,
            paradas=paradas,
            hora_saida=hora_saida,
        )

        salvo_neo4j = False
        if cooperativa_id and self.rota_repository is not None:
            try:
                info_salvo = self.rota_repository.salvar_rota_otimizada(
                    resultado=resultado,
                    origem=origem,
                    paradas_entrada=paradas,
                    cooperativa_id=cooperativa_id,
                    rota_id=rota_id_neo4j,
                    nome_rota=nome_rota_neo4j,
                )
                rota_id_neo4j = info_salvo.get('rota_id')
                salvo_neo4j = True
            except Exception as exc:
                log.error(f'[logistica] Falha ao persistir rota otimizada no Neo4j: {exc}')
                salvo_neo4j = False

        passos_fmt = [
            f'{p.ordem}º {p.endereco} ({p.zona} — {p.distancia_trecho_km:.1f} km, ~{p.tempo_trecho_min:.0f} min)'
            for p in resultado.paradas_ordenadas
        ]
        sequencia_str = ' -> '.join(passos_fmt)

        economia_txt = (
            f'economia estimada de {resultado.economia_estimada_min:.1f} min ({resultado.economia_percentual:.1f}%)'
            if resultado.economia_estimada_min > 0
            else 'a sequência informada já é a mais rápida para o horário'
        )

        resposta_sintese = (
            f'Rota otimizada saindo de {resultado.origem} às {resultado.hora_saida}h: '
            f'{sequencia_str}. Distância viária total: {resultado.distancia_total_km:.1f} km; '
            f'tempo total estimado pelo modelo supervisionado: {resultado.tempo_total_estimado_min:.1f} min '
            f'({economia_txt}).'
        )

        recomendacao_operacional = (
            'Siga a ordem otimizada para minimizar cruzamentos de zona e retenção viária em São Paulo.'
            if not (7 <= resultado.hora_saida <= 9 or 17 <= resultado.hora_saida <= 19)
            else (
                'Atenção: saída em horário de pico em São Paulo; '
                'priorize as paradas da mesma macrozona antes de cruzar o Centro ou Marginais.'
            )
        )

        payload_ok = {
            'dominio': 'logistica',
            'intencao': 'otimizar_rota',
            'resposta': resposta_sintese,
            'recomendacao': recomendacao_operacional,
            'indicadores': {
                'qtd_paradas': len(resultado.paradas_ordenadas),
                'distancia_total_km': resultado.distancia_total_km,
                'tempo_total_estimado_min': resultado.tempo_total_estimado_min,
                'tempo_ordem_original_min': resultado.tempo_ordem_original_min,
                'economia_estimada_min': resultado.economia_estimada_min,
                'economia_percentual': resultado.economia_percentual,
                'hora_saida': resultado.hora_saida,
                'salvo_no_neo4j': salvo_neo4j,
            },
        }
        return self._empacotar_saida(payload_ok)

    def _empacotar_saida(self, payload: dict[str, Any]) -> dict[str, Any]:
        conteudo_json = json.dumps(payload, ensure_ascii=False)
        return {
            'messages': [{'role': 'assistant', 'content': conteudo_json}],
            'agentes_chamados': [self.nome_agente],
            'especialista_outputs': {
                self.nome_agente: EspecialistaOutput(
                    conteudo=conteudo_json,
                    fonte=self.nome_agente,
                )
            },
        }

    def run(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        return self._executar_pipeline(state, config)

    async def arun(self, state: GraphState, config: RunnableConfig | None = None) -> dict:
        return await asyncio.to_thread(self._executar_pipeline, state, config)
