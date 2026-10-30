from collections import defaultdict
from datetime import datetime, time, timedelta
from decimal import Decimal
from typing import Optional, List
from uuid import UUID
import unicodedata

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.base import Toolkit
from app.repository.postgresql.coleta_repository import ColetaRepository, ColetaSnapshot, Coleta
from app.repository.postgresql.triagem_repository import TriagemRepository, TriagemSnapshot, Triagem
from app.repository.postgresql.funcionario_repository import FuncionarioRepository, FuncionarioSnapshot, Funcionario
from app.repository.postgresql.equipe_repository import EquipeRepository, EquipeSnapshot, Equipe
from app.repository.postgresql.material_repository import MaterialRepository, MaterialSnapshot


class ConsultarColetasArgs(BaseModel):
    coleta_ids: Optional[List[UUID]] = Field(
        default=None,
        description='Lista opcional de UUIDs de coletas específicas para filtrar.'
    )
    status: Optional[str] = Field(
        default=None,
        description="Filtra pelo status atual da coleta (ex: 'AGENDADA', 'PENDENTE', 'CONCLUIDA', 'CANCELADA')."
    )
    tipo_coleta: Optional[str] = Field(
        default=None,
        description="Filtra pelo tipo de coleta ('EXTERNA' ou 'ENTREGA')."
    )
    cooperado_id: Optional[UUID] = Field(
        default=None,
        description='UUID do cooperado/funcionário responsável pela coleta.'
    )
    cooperado_nome: Optional[str] = Field(
        default=None,
        description='Nome (ou parte do nome) do cooperado responsável pela coleta.'
    )
    equipe_id: Optional[UUID] = Field(
        default=None,
        description='UUID da equipe (filtra coletas realizadas por membros da equipe ou triadas pela equipe).'
    )
    equipe_nome: Optional[str] = Field(
        default=None,
        description='Nome (ou parte do nome) da equipe responsável.'
    )
    rota_id: Optional[UUID] = Field(
        default=None,
        description='UUID da rota associada à coleta.'
    )
    data_inicio: Optional[datetime] = Field(
        default=None,
        description='Data inicial do intervalo de busca (ISO 8601).'
    )
    data_fim: Optional[datetime] = Field(
        default=None,
        description='Data final do intervalo de busca (ISO 8601).'
    )
    quantidade_min_kg: Optional[float] = Field(
        default=None,
        description='Quantidade mínima coletada (kg).'
    )
    quantidade_max_kg: Optional[float] = Field(
        default=None,
        description='Quantidade máxima coletada (kg).'
    )
    limite: int = Field(default=20, description='Número máximo de coletas detalhadas retornadas.')


class ConsultarTriagensArgs(BaseModel):
    triagem_ids: Optional[List[UUID]] = Field(
        default=None,
        description='Lista opcional de UUIDs de triagens específicas.'
    )
    coleta_id: Optional[UUID] = Field(
        default=None,
        description='UUID da coleta de origem para listar todas as N triagens daquela coleta.'
    )
    material_ids: Optional[List[UUID]] = Field(
        default=None,
        description='Lista de UUIDs de materiais para filtrar triagens.'
    )
    material_nomes: Optional[List[str]] = Field(
        default=None,
        description="Nome (ou parte do nome) de um ou mais materiais/categorias (ex: ['PET', 'papelão', 'plástico'])."
    )
    equipe_id: Optional[UUID] = Field(
        default=None,
        description='UUID da equipe responsável pela triagem.'
    )
    equipe_nome: Optional[str] = Field(
        default=None,
        description='Nome (ou parte do nome) da equipe de triagem.'
    )
    status: Optional[str] = Field(
        default=None,
        description="Filtra pelo status atual da triagem (ex: 'EM_ANDAMENTO', 'CONCLUIDA')."
    )
    data_inicio: Optional[datetime] = Field(
        default=None,
        description='Data inicial do intervalo de triagem (ISO 8601).'
    )
    data_fim: Optional[datetime] = Field(
        default=None,
        description='Data final do intervalo de triagem (ISO 8601).'
    )
    limite: int = Field(default=20, description='Número máximo de triagens detalhadas retornadas.')


class ConsultarEquipesCooperadosArgs(BaseModel):
    equipe_id: Optional[UUID] = Field(
        default=None,
        description='UUID de uma equipe específica.'
    )
    equipe_nome: Optional[str] = Field(
        default=None,
        description='Nome (ou parte do nome) da equipe.'
    )
    cooperado_id: Optional[UUID] = Field(
        default=None,
        description='UUID de um funcionário/cooperado específico.'
    )
    cooperado_nome: Optional[str] = Field(
        default=None,
        description='Nome (ou parte do nome) do cooperado ou gestor.'
    )
    cargo: Optional[str] = Field(
        default=None,
        description="Filtra funcionários pelo nome do cargo (ex: 'Cooperado', 'Gestor', 'Motorista')."
    )
    apenas_ativos: bool = Field(
        default=True,
        description='Se true, retorna apenas equipes e funcionários ativos.'
    )


class ResumoOperacionalArgs(BaseModel):
    data_inicio: Optional[datetime] = Field(
        default=None,
        description='Data inicial opcional para restringir o resumo operacional (ISO 8601).'
    )
    data_fim: Optional[datetime] = Field(
        default=None,
        description='Data final opcional para restringir o resumo operacional (ISO 8601).'
    )


class ColetaToolkit(Toolkit):
    nome: str = 'coleta_toolkit'
    descricao: str = 'Ferramentas para consulta de coletas, triagens, equipes e cooperados de cooperativas.'
    repositories = {
        'coleta_repository': ColetaRepository,
        'triagem_repository': TriagemRepository,
        'funcionario_repository': FuncionarioRepository,
        'equipe_repository': EquipeRepository,
        'material_repository': MaterialRepository,
    }

    def __init__(self, repositories=None):
        self.repositories = repositories or self.__class__.repositories

    def _normalizar(self, texto: str) -> str:
        """Remove acentos, underline/hífen e baixa a caixa para comparação tolerante."""
        sem_acento = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('ascii')
        return sem_acento.lower().replace('_', ' ').replace('-', ' ').strip()

    def _corresponde_nome(self, termo: str, alvo: str) -> bool:
        """Verifica se termo de busca bate com o nome alvo com tolerância a plurais/substrings."""
        if not termo or not alvo:
            return False
        if termo in alvo or alvo in termo:
            return True
        if termo.endswith('s') and termo[:-1] in alvo:
            return True
        if alvo.endswith('s') and alvo[:-1] in termo:
            return True
        return False

    def _resolver_intervalo_data(
        self,
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
    ) -> tuple[Optional[datetime], Optional[datetime]]:
        if data_inicio is not None and data_inicio.time() == time(0, 0):
            data_inicio = datetime.combine(data_inicio.date(), time.min)
        if data_fim is not None and data_fim.time() == time(0, 0):
            data_fim = datetime.combine(data_fim.date() + timedelta(days=1), time.min)
        return data_inicio, data_fim

    def _resolver_materiais_ids(
        self,
        material_snapshot: Optional[MaterialSnapshot],
        material_ids: Optional[List[UUID]],
        material_nomes: Optional[List[str]],
    ) -> Optional[set[UUID]]:
        """Retorna o conjunto de material_ids candidatos quando material_snapshot está disponível."""
        if not material_ids and not material_nomes:
            return None
        if material_snapshot is None:
            return set(material_ids) if material_ids else None

        ids_set = set(material_ids or [])
        termos = [self._normalizar(n) for n in (material_nomes or []) if n]

        resolvidos: set[UUID] = set()
        for item in material_snapshot.itens:
            if item.material_id in ids_set:
                resolvidos.add(item.material_id)
                continue
            nome_cat = self._normalizar(item.nome_categoria or '')
            nome_pai = self._normalizar(item.nome_categoria_pai or '')
            if any(
                self._corresponde_nome(termo, nome_cat) or self._corresponde_nome(termo, nome_pai)
                for termo in termos
            ):
                resolvidos.add(item.material_id)
        return resolvidos

    @Toolkit.timed(categoria='tool', nome='consultar_coletas')
    def consultar_coletas(
        self,
        config: RunnableConfig,
        coleta_ids: Optional[List[UUID]] = None,
        status: Optional[str] = None,
        tipo_coleta: Optional[str] = None,
        cooperado_id: Optional[UUID] = None,
        cooperado_nome: Optional[str] = None,
        equipe_id: Optional[UUID] = None,
        equipe_nome: Optional[str] = None,
        rota_id: Optional[UUID] = None,
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        limite: int = 20,
    ) -> dict:
        """Consulta registros de coletas da cooperativa e cruza com as triagens associadas."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        coleta_snapshot: Optional[ColetaSnapshot] = snapshots.get('coleta_snapshot')
        triagem_snapshot: Optional[TriagemSnapshot] = snapshots.get('triagem_snapshot')
        funcionario_snapshot: Optional[FuncionarioSnapshot] = snapshots.get('funcionario_snapshot')
        equipe_snapshot: Optional[EquipeSnapshot] = snapshots.get('equipe_snapshot')

        if not coleta_snapshot:
            return {'status': 'error', 'message': 'Snapshot de coletas indisponível para esta cooperativa.'}

        funcionarios_por_id: dict[UUID, Funcionario] = (
            {f.funcionario_id: f for f in funcionario_snapshot.itens}
            if funcionario_snapshot else {}
        )

        triagens_por_coleta: dict[UUID, list[Triagem]] = defaultdict(list)
        if triagem_snapshot:
            for t in triagem_snapshot.itens:
                triagens_por_coleta[t.coleta_id].append(t)

        # Resolve filtro de equipe (membros da equipe ou triagens realizadas pela equipe)
        cooperados_da_equipe: set[UUID] = set()
        coletas_triadas_pela_equipe: set[UUID] = set()
        filtrar_equipe = bool(equipe_id or equipe_nome)
        if filtrar_equipe:
            termo_equipe = self._normalizar(equipe_nome) if equipe_nome else None
            equipes_alvo: set[UUID] = set()
            if equipe_id:
                equipes_alvo.add(equipe_id)
            if equipe_snapshot and termo_equipe:
                for eq in equipe_snapshot.itens:
                    if self._corresponde_nome(termo_equipe, self._normalizar(eq.nome or '')):
                        equipes_alvo.add(eq.equipe_id)
            if equipe_snapshot:
                for eq in equipe_snapshot.itens:
                    if eq.equipe_id in equipes_alvo:
                        cooperados_da_equipe.update(eq.cooperado_ids)
                        cooperados_da_equipe.add(eq.gestor_id)
            if triagem_snapshot:
                for t in triagem_snapshot.itens:
                    if t.equipe_id in equipes_alvo or (
                        termo_equipe and self._corresponde_nome(termo_equipe, self._normalizar(t.nome_equipe or ''))
                    ):
                        coletas_triadas_pela_equipe.add(t.coleta_id)

        data_ini_norm, data_fim_norm = self._resolver_intervalo_data(data_inicio, data_fim)
        coleta_ids_set = set(coleta_ids) if coleta_ids else None
        termo_coop = self._normalizar(cooperado_nome) if cooperado_nome else None

        itens_filtrados: list[Coleta] = []
        for c in coleta_snapshot.itens:
            if coleta_ids_set and c.coleta_id not in coleta_ids_set:
                continue
            if status and (not c.status_atual or c.status_atual.strip().upper() != status.strip().upper()):
                continue
            if tipo_coleta and (not c.tipo_coleta or c.tipo_coleta.strip().upper() != tipo_coleta.strip().upper()):
                continue
            if cooperado_id and c.cooperado_id != cooperado_id:
                continue
            if termo_coop:
                nome_coop = c.cooperado_responsavel or (
                    funcionarios_por_id[c.cooperado_id].nome if c.cooperado_id in funcionarios_por_id else ''
                )
                if not self._corresponde_nome(termo_coop, self._normalizar(nome_coop)):
                    continue
            if filtrar_equipe:
                pertence_equipe = (
                    c.cooperado_id in cooperados_da_equipe
                    or c.coleta_id in coletas_triadas_pela_equipe
                )
                if not pertence_equipe:
                    continue
            if rota_id and c.rota_id != rota_id:
                continue
            if data_ini_norm and c.data_coleta < data_ini_norm:
                continue
            if data_fim_norm and c.data_coleta >= data_fim_norm:
                continue
            if quantidade_min_kg is not None and c.quantidade_kg < Decimal(str(quantidade_min_kg)):
                continue
            if quantidade_max_kg is not None and c.quantidade_kg > Decimal(str(quantidade_max_kg)):
                continue

            itens_filtrados.append(c)

        itens_retornados = itens_filtrados[:limite]
        peso_total_coletado = sum((c.quantidade_kg or Decimal(0)) for c in itens_filtrados)

        coletas_detalhadas = []
        for c in itens_retornados:
            triagens_c = triagens_por_coleta.get(c.coleta_id, [])
            peso_triado_c = sum((t.quantidade_kg or Decimal(0)) for t in triagens_c)
            peso_rejeito_c = sum((t.quantidade_rejeito_kg or Decimal(0)) for t in triagens_c)
            materiais_c = sorted({t.nome_categoria for t in triagens_c if t.nome_categoria})
            nome_coop = c.cooperado_responsavel or (
                funcionarios_por_id[c.cooperado_id].nome if c.cooperado_id in funcionarios_por_id else None
            )

            coletas_detalhadas.append({
                'coleta_id': str(c.coleta_id),
                'cooperado_id': str(c.cooperado_id),
                'cooperado_responsavel': nome_coop,
                'rota_id': str(c.rota_id) if c.rota_id else None,
                'data_coleta': c.data_coleta.isoformat() if c.data_coleta else None,
                'status_atual': c.status_atual,
                'tipo_coleta': c.tipo_coleta,
                'quantidade_kg': float(c.quantidade_kg or 0),
                'total_triagens_vinculadas': len(triagens_c),
                'peso_triado_kg': float(peso_triado_c),
                'peso_rejeito_kg': float(peso_rejeito_c),
                'materiais_triados': materiais_c,
            })

        return {
            'status': 'ok',
            'total_coletas_encontradas': len(itens_filtrados),
            'peso_total_coletado_kg': float(peso_total_coletado),
            'coletas': coletas_detalhadas,
        }

    @Toolkit.timed(categoria='tool', nome='consultar_triagens')
    def consultar_triagens(
        self,
        config: RunnableConfig,
        triagem_ids: Optional[List[UUID]] = None,
        coleta_id: Optional[UUID] = None,
        material_ids: Optional[List[UUID]] = None,
        material_nomes: Optional[List[str]] = None,
        equipe_id: Optional[UUID] = None,
        equipe_nome: Optional[str] = None,
        status: Optional[str] = None,
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
        limite: int = 20,
    ) -> dict:
        """Consulta registros de triagens e separação de materiais por categoria, coleta ou equipe."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        triagem_snapshot: Optional[TriagemSnapshot] = snapshots.get('triagem_snapshot')
        equipe_snapshot: Optional[EquipeSnapshot] = snapshots.get('equipe_snapshot')
        material_snapshot: Optional[MaterialSnapshot] = snapshots.get('material_snapshot')

        if not triagem_snapshot:
            return {'status': 'error', 'message': 'Snapshot de triagens indisponível para esta cooperativa.'}

        equipes_por_id: dict[UUID, Equipe] = (
            {e.equipe_id: e for e in equipe_snapshot.itens}
            if equipe_snapshot else {}
        )
        materiais_por_id = (
            {m.material_id: m for m in material_snapshot.itens}
            if material_snapshot else {}
        )

        filtrar_material = bool(material_ids or material_nomes)
        materiais_candidatos_ids = self._resolver_materiais_ids(material_snapshot, material_ids, material_nomes)
        termos_material = [self._normalizar(n) for n in (material_nomes or []) if n]

        triagem_ids_set = set(triagem_ids) if triagem_ids else None
        termo_equipe = self._normalizar(equipe_nome) if equipe_nome else None
        data_ini_norm, data_fim_norm = self._resolver_intervalo_data(data_inicio, data_fim)

        triagens_filtradas: list[Triagem] = []
        for t in triagem_snapshot.itens:
            if triagem_ids_set and t.triagem_id not in triagem_ids_set:
                continue
            if coleta_id and t.coleta_id != coleta_id:
                continue
            if status and (not t.status_atual or t.status_atual.strip().upper() != status.strip().upper()):
                continue
            if equipe_id and t.equipe_id != equipe_id:
                continue
            if termo_equipe:
                nome_eq = t.nome_equipe or (
                    equipes_por_id[t.equipe_id].nome if t.equipe_id in equipes_por_id else ''
                )
                if not self._corresponde_nome(termo_equipe, self._normalizar(nome_eq or '')):
                    continue
            if filtrar_material:
                bateu_id = materiais_candidatos_ids is not None and t.material_id in materiais_candidatos_ids
                nome_cat = self._normalizar(t.nome_categoria or '')
                nome_pai = self._normalizar(t.nome_categoria_pai or '')
                bateu_nome_direto = any(
                    self._corresponde_nome(termo, nome_cat) or self._corresponde_nome(termo, nome_pai)
                    for termo in termos_material
                )
                if not (bateu_id or bateu_nome_direto):
                    continue
            if data_ini_norm and t.data_triagem < data_ini_norm:
                continue
            if data_fim_norm and t.data_triagem >= data_fim_norm:
                continue

            triagens_filtradas.append(t)

        itens_retornados = triagens_filtradas[:limite]
        peso_total_triado = sum((t.quantidade_kg or Decimal(0)) for t in triagens_filtradas)
        peso_total_rejeito = sum((t.quantidade_rejeito_kg or Decimal(0)) for t in triagens_filtradas)
        volume_bruto_triagem = peso_total_triado + peso_total_rejeito
        taxa_rejeito = (
            float(peso_total_rejeito / volume_bruto_triagem * 100)
            if volume_bruto_triagem > 0 else 0.0
        )

        quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
        for t in triagens_filtradas:
            info_mat = materiais_por_id.get(t.material_id)
            cat_nome = t.nome_categoria or (info_mat.nome_categoria if info_mat else 'Sem Categoria')
            quantidade_por_categoria[cat_nome] += (t.quantidade_kg or Decimal(0))

        triagens_detalhadas = []
        for t in itens_retornados:
            info_mat = materiais_por_id.get(t.material_id)
            nome_eq = t.nome_equipe or (
                equipes_por_id[t.equipe_id].nome if t.equipe_id in equipes_por_id else None
            )
            cat_nome = t.nome_categoria or (info_mat.nome_categoria if info_mat else None)
            cat_pai_nome = t.nome_categoria_pai or (info_mat.nome_categoria_pai if info_mat else None)

            triagens_detalhadas.append({
                'triagem_id': str(t.triagem_id),
                'coleta_id': str(t.coleta_id),
                'equipe_id': str(t.equipe_id),
                'nome_equipe': nome_eq,
                'material_id': str(t.material_id),
                'nome_categoria': cat_nome,
                'nome_categoria_pai': cat_pai_nome,
                'data_triagem': t.data_triagem.isoformat() if t.data_triagem else None,
                'status_atual': t.status_atual,
                'quantidade_kg': float(t.quantidade_kg or 0),
                'quantidade_rejeito_kg': float(t.quantidade_rejeito_kg or 0),
            })

        return {
            'status': 'ok',
            'total_triagens_encontradas': len(triagens_filtradas),
            'peso_total_triado_kg': float(peso_total_triado),
            'peso_total_rejeito_kg': float(peso_total_rejeito),
            'taxa_rejeito_percentual': round(taxa_rejeito, 2),
            'quantidade_por_categoria_kg': {k: float(v) for k, v in quantidade_por_categoria.items()},
            'triagens': triagens_detalhadas,
        }

    @Toolkit.timed(categoria='tool', nome='consultar_equipes_cooperados')
    def consultar_equipes_cooperados(
        self,
        config: RunnableConfig,
        equipe_id: Optional[UUID] = None,
        equipe_nome: Optional[str] = None,
        cooperado_id: Optional[UUID] = None,
        cooperado_nome: Optional[str] = None,
        cargo: Optional[str] = None,
        apenas_ativos: bool = True,
    ) -> dict:
        """Consulta as equipes operacionais da cooperativa, seus gestores e os cooperados vinculados."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        equipe_snapshot: Optional[EquipeSnapshot] = snapshots.get('equipe_snapshot')
        funcionario_snapshot: Optional[FuncionarioSnapshot] = snapshots.get('funcionario_snapshot')

        if not equipe_snapshot and not funcionario_snapshot:
            return {
                'status': 'error',
                'message': 'Snapshots de equipes e funcionários indisponíveis para esta cooperativa.',
            }

        funcionarios_por_id: dict[UUID, Funcionario] = (
            {f.funcionario_id: f for f in funcionario_snapshot.itens}
            if funcionario_snapshot else {}
        )

        termo_equipe = self._normalizar(equipe_nome) if equipe_nome else None
        termo_coop = self._normalizar(cooperado_nome) if cooperado_nome else None
        termo_cargo = self._normalizar(cargo) if cargo else None

        equipes_detalhadas = []
        if equipe_snapshot:
            for eq in equipe_snapshot.itens:
                if apenas_ativos and not eq.esta_ativa:
                    continue
                if equipe_id and eq.equipe_id != equipe_id:
                    continue
                if termo_equipe and not self._corresponde_nome(termo_equipe, self._normalizar(eq.nome or '')):
                    continue
                if cooperado_id and cooperado_id not in eq.cooperado_ids and eq.gestor_id != cooperado_id:
                    continue
                if termo_coop:
                    nomes_membros = [
                        self._normalizar(funcionarios_por_id[cid].nome)
                        for cid in eq.cooperado_ids
                        if cid in funcionarios_por_id
                    ]
                    if eq.nome_gestor:
                        nomes_membros.append(self._normalizar(eq.nome_gestor))
                    if not any(self._corresponde_nome(termo_coop, nm) for nm in nomes_membros):
                        continue

                membros = []
                for cid in eq.cooperado_ids:
                    f = funcionarios_por_id.get(cid)
                    if f:
                        membros.append({
                            'funcionario_id': str(f.funcionario_id),
                            'nome': f.nome,
                            'cargo': f.cargo,
                            'status_funcionario': f.status_funcionario,
                        })
                    else:
                        membros.append({'funcionario_id': str(cid)})

                equipes_detalhadas.append({
                    'equipe_id': str(eq.equipe_id),
                    'nome': eq.nome,
                    'descricao': eq.descricao,
                    'esta_ativa': eq.esta_ativa,
                    'gestor_id': str(eq.gestor_id),
                    'nome_gestor': eq.nome_gestor,
                    'numero_cooperados': len(eq.cooperado_ids),
                    'cooperados': membros,
                })

        funcionarios_detalhados = []
        if funcionario_snapshot:
            for f in funcionario_snapshot.itens:
                if apenas_ativos and not f.esta_ativo:
                    continue
                if cooperado_id and f.funcionario_id != cooperado_id:
                    continue
                if termo_coop and not self._corresponde_nome(termo_coop, self._normalizar(f.nome)):
                    continue
                if termo_cargo and not self._corresponde_nome(termo_cargo, self._normalizar(f.cargo)):
                    continue
                funcionarios_detalhados.append({
                    'funcionario_id': str(f.funcionario_id),
                    'nome': f.nome,
                    'cargo': f.cargo,
                    'status_funcionario': f.status_funcionario,
                    'esta_ativo': f.esta_ativo,
                    'esta_afastado': f.esta_afastado,
                    'data_admissao': f.data_admissao.isoformat() if f.data_admissao else None,
                })

        return {
            'status': 'ok',
            'total_equipes': len(equipes_detalhadas),
            'total_funcionarios': len(funcionarios_detalhados),
            'equipes': equipes_detalhadas,
            'funcionarios': funcionarios_detalhados,
        }

    @Toolkit.timed(categoria='tool', nome='resumo_operacional')
    def resumo_operacional(
        self,
        config: RunnableConfig,
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
    ) -> dict:
        """Gera um resumo integrado das operações de coleta, triagem, materiais triados e equipes da cooperativa."""
        snapshots = ((config or {}).get('configurable') or {}).get('snapshots') or {}
        coleta_snapshot: Optional[ColetaSnapshot] = snapshots.get('coleta_snapshot')
        triagem_snapshot: Optional[TriagemSnapshot] = snapshots.get('triagem_snapshot')
        equipe_snapshot: Optional[EquipeSnapshot] = snapshots.get('equipe_snapshot')
        funcionario_snapshot: Optional[FuncionarioSnapshot] = snapshots.get('funcionario_snapshot')
        material_snapshot: Optional[MaterialSnapshot] = snapshots.get('material_snapshot')

        if not coleta_snapshot and not triagem_snapshot:
            return {'status': 'error', 'message': 'Snapshots operacionais de coleta e triagem indisponíveis.'}

        data_ini_norm, data_fim_norm = self._resolver_intervalo_data(data_inicio, data_fim)

        coletas = list(coleta_snapshot.itens) if coleta_snapshot else []
        if data_ini_norm:
            coletas = [c for c in coletas if c.data_coleta >= data_ini_norm]
        if data_fim_norm:
            coletas = [c for c in coletas if c.data_coleta < data_fim_norm]

        triagens = list(triagem_snapshot.itens) if triagem_snapshot else []
        if data_ini_norm:
            triagens = [t for t in triagens if t.data_triagem >= data_ini_norm]
        if data_fim_norm:
            triagens = [t for t in triagens if t.data_triagem < data_fim_norm]

        materiais_por_id = (
            {m.material_id: m for m in material_snapshot.itens}
            if material_snapshot else {}
        )

        total_coletas = len(coletas)
        total_triagens = len(triagens)
        peso_coletas = sum((c.quantidade_kg or Decimal(0)) for c in coletas)
        peso_triagens = sum((t.quantidade_kg or Decimal(0)) for t in triagens)
        peso_rejeito = sum((t.quantidade_rejeito_kg or Decimal(0)) for t in triagens)
        volume_bruto_triado = peso_triagens + peso_rejeito

        taxa_rejeito = (
            round(float(peso_rejeito / volume_bruto_triado * 100), 2)
            if volume_bruto_triado > 0 else 0.0
        )
        taxa_aproveitamento = (
            round(float(peso_triagens / volume_bruto_triado * 100), 2)
            if volume_bruto_triado > 0 else 100.0
        )

        quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
        for t in triagens:
            info_mat = materiais_por_id.get(t.material_id)
            cat_nome = t.nome_categoria or (info_mat.nome_categoria if info_mat else 'Sem Categoria')
            quantidade_por_categoria[cat_nome] += (t.quantidade_kg or Decimal(0))

        equipes_ativas = (
            sum(1 for e in equipe_snapshot.itens if e.esta_ativa)
            if equipe_snapshot else 0
        )
        cooperados_ativos = (
            sum(1 for f in funcionario_snapshot.itens if f.esta_ativo)
            if funcionario_snapshot else 0
        )

        return {
            'status': 'ok',
            'total_coletas': total_coletas,
            'peso_coletado_kg': float(peso_coletas),
            'total_triagens': total_triagens,
            'peso_triado_kg': float(peso_triagens),
            'peso_rejeito_kg': float(peso_rejeito),
            'taxa_rejeito_percentual': taxa_rejeito,
            'taxa_aproveitamento_percentual': taxa_aproveitamento,
            'quantidade_por_categoria_kg': {k: float(v) for k, v in quantidade_por_categoria.items()},
            'equipes_ativas': equipes_ativas,
            'cooperados_ativos': cooperados_ativos,
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.consultar_coletas,
                name='consultar_coletas',
                description=self.consultar_coletas.__doc__,
                args_schema=ConsultarColetasArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_triagens,
                name='consultar_triagens',
                description=self.consultar_triagens.__doc__,
                args_schema=ConsultarTriagensArgs,
            ),
            StructuredTool.from_function(
                func=self.consultar_equipes_cooperados,
                name='consultar_equipes_cooperados',
                description=self.consultar_equipes_cooperados.__doc__,
                args_schema=ConsultarEquipesCooperadosArgs,
            ),
            StructuredTool.from_function(
                func=self.resumo_operacional,
                name='resumo_operacional',
                description=self.resumo_operacional.__doc__,
                args_schema=ResumoOperacionalArgs,
            ),
        ]
