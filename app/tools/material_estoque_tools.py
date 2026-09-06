from app.tools.base import Toolkit
from app.repository.postgresql.estoque_repository import EstoqueRepository, EstoqueSnapshot
from app.repository.postgresql.material_repository import MaterialRepository, MaterialSnapshot

from collections import defaultdict
from pydantic import BaseModel, Field
from decimal import Decimal
from uuid import UUID
from typing import Optional, List

import unicodedata
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool


class ConsultarEstoqueArgs(BaseModel):
    material_ids: Optional[List[UUID]] = Field(
        default=None,
        description="IDs em materiais (FK). Pode conter um ou mais materiais."
    )
    material_names: Optional[List[str]] = Field(
        default=None,
        description="Nome (ou parte do nome) de um ou mais materiais, para busca quando o ID não é conhecido. Ex: ['PET', 'papelão']."
    )
    disponivel: Optional[bool] = Field(
        default=None,
        description="Filtra por disponibilidade do material no snapshot (true = apenas disponíveis). Ausente = não filtra por disponibilidade."
    )
    quantidade_min_kg: Optional[float] = Field(
        default=None,
        description="Quantidade mínima em estoque (kg) para incluir o material no resultado. Útil para 'o que está acabando' (ex: quantidade_max_kg baixo) ou 'o que tem bastante' (quantidade_min_kg alto)."
    )
    quantidade_max_kg: Optional[float] = Field(
        default=None,
        description="Quantidade máxima em estoque (kg) para incluir o material no resultado."
    )
    preco_min: Optional[float] = Field(default=None, description="Preço sugerido mínimo do material.")
    preco_max: Optional[float] = Field(default=None, description="Preço sugerido máximo do material.")
    incluir_sem_estoque: bool = Field(
        default=False,
        description="Se true, inclui materiais com quantidade em estoque igual a zero no resultado (por padrão, ficam de fora)."
    )


class GeneralizarEstoqueArgs(BaseModel):
    material_ids: Optional[List[UUID]] = Field(
        default=None,
        description="IDs em materiais (FK), pra restringir a materiais específicos antes de agrupar pela categoria-pai."
    )
    material_names: Optional[List[str]] = Field(
        default=None,
        description="Nome (ou parte do nome) de um ou mais materiais, quando o ID não é conhecido. Ex: ['PET', 'papelão']."
    )
    categoria_ids: Optional[List[UUID]] = Field(
        default=None,
        description=(
            "FK em categorias_materiais. Aceita tanto o id de uma categoria-pai (ex: 'Plástico') "
            "quanto de uma categoria-filha (ex: 'PET') — nos dois casos, o resultado é agrupado "
            "pela categoria-pai correspondente. Ausente = agrupa todos os materiais disponíveis."
        )
    )
    disponivel: Optional[bool] = Field(default=None, description="Filtra por disponibilidade (true = apenas disponíveis).")
    quantidade_min_kg: Optional[float] = Field(default=None, description="Quantidade mínima em estoque (kg).")
    quantidade_max_kg: Optional[float] = Field(default=None, description="Quantidade máxima em estoque (kg).")
    preco_min: Optional[float] = Field(default=None, description="Preço sugerido mínimo do material.")
    preco_max: Optional[float] = Field(default=None, description="Preço sugerido máximo do material.")
    incluir_sem_estoque: bool = Field(default=False, description="Se true, inclui materiais com estoque zerado.")


class MaterialEstoqueToolkit(Toolkit):
    nome = 'material_estoque_toolkit'
    descricao = 'Tools de consulta de estoque e materiais do Renovaí.'
    repositories = {
        'estoque_repository':  EstoqueRepository,
        'material_repository': MaterialRepository,
    }

    _MATERIAL_ALIASES: dict[str, str] = {}

    def _normalizar(self, texto: str) -> str:
        """Remove acentos, underline/hífen e baixa a caixa, pra comparação tolerante."""
        sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
        return sem_acento.lower().replace("_", " ").replace("-", " ").strip()

    def _resolver_materiais(
        self,
        material_snapshot: MaterialSnapshot,
        material_ids: Optional[List[UUID]],
        material_names: Optional[List[str]],
    ) -> list:
        """Resolve a lista de materiais candidatos a partir de ids e/ou nomes (OR dentro do campo).
        Sem nenhum dos dois filtros, retorna todos os materiais do snapshot."""
        if not material_ids and not material_names:
            return list(material_snapshot.itens)

        ids_set = set(material_ids or [])
        termos = [
            self._MATERIAL_ALIASES.get(self._normalizar(n).upper(), self._normalizar(n))
            for n in (material_names or [])
        ]

        return [
            item for item in material_snapshot.itens
            if item.material_id in ids_set
            or any(termo in self._normalizar(item.nome_categoria) for termo in termos)
        ]

    def _resolver_categoria_pai_estoque(self, estoque_item) -> tuple[UUID, str]:
        """Numa árvore de 2 níveis, a categoria-pai efetiva de um item de estoque é ele mesmo
        quando `categoria_pai_id`/`nome_categoria_pai` são None (o item já é uma categoria raiz).

        getattr com default None só protege contra o atributo não existir ainda no objeto
        (ex: repository desatualizado/skew de deploy) — não é proteção contra dado errado,
        porque não tem como esse helper saber se um None é 'raiz de verdade' ou 'join quebrado'.
        """
        categoria_pai_id = getattr(estoque_item, "categoria_pai_id", None)
        nome_categoria_pai = getattr(estoque_item, "nome_categoria_pai", None)

        cat_pai_id = categoria_pai_id if categoria_pai_id is not None else estoque_item.categoria_id
        nome_cat_pai = nome_categoria_pai if nome_categoria_pai is not None else estoque_item.nome_categoria

        return cat_pai_id, nome_cat_pai

    def _passa_filtros(
        self,
        estoque_item,
        info_material,
        disponivel: Optional[bool],
        quantidade_min_kg: Optional[float],
        quantidade_max_kg: Optional[float],
        preco_min: Optional[float],
        preco_max: Optional[float],
        incluir_sem_estoque: bool,
    ) -> bool:
        if not incluir_sem_estoque and estoque_item.quantidade_kg == 0:
            return False

        # quantidade_kg/preco_sugerido são Decimal, usando cast para evitar TypeError comparando com float
        if quantidade_min_kg is not None and estoque_item.quantidade_kg < Decimal(str(quantidade_min_kg)):
            return False
        if quantidade_max_kg is not None and estoque_item.quantidade_kg > Decimal(str(quantidade_max_kg)):
            return False

        if disponivel is not None:
            if info_material is None or info_material.esta_disponivel != disponivel:
                return False

        if info_material and info_material.preco_sugerido is not None:
            preco = info_material.preco_sugerido
            if preco_min is not None and preco < Decimal(str(preco_min)):
                return False
            if preco_max is not None and preco > Decimal(str(preco_max)):
                return False

        return True

    def consultar_estoque(
        self,
        config: RunnableConfig,
        material_ids: Optional[List[UUID]] = None,
        material_names: Optional[List[str]] = None,
        disponivel: Optional[bool] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        preco_min: Optional[float] = None,
        preco_max: Optional[float] = None,
        incluir_sem_estoque: bool = False,
    ) -> dict:
        """Consulta o estoque disponível de um ou mais materiais específicos, com filtros por
        disponibilidade, faixa de quantidade (kg) e faixa de preço sugerido."""
        print('[DEBUG]: acessou tool - consultar_estoque')

        snapshots = config['configurable']['snapshots']
        estoque_snapshot: EstoqueSnapshot = snapshots['estoque_snapshot']
        material_snapshot: MaterialSnapshot = snapshots['material_snapshot']

        materiais_candidatos = self._resolver_materiais(material_snapshot, material_ids, material_names)
        if not materiais_candidatos:
            return {'status': 'error', 'message': 'Nenhum material correspondente encontrado.'}

        material_por_id = {item.material_id: item for item in materiais_candidatos}

        quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
        detalhes = []
        for estoque_item in estoque_snapshot.itens:
            info_material = material_por_id.get(estoque_item.material_id)
            if info_material is None:
                continue

            if not self._passa_filtros(
                estoque_item, info_material,
                disponivel, quantidade_min_kg, quantidade_max_kg,
                preco_min, preco_max, incluir_sem_estoque,
            ):
                continue

            quantidade_por_categoria[estoque_item.nome_categoria] += estoque_item.quantidade_kg
            detalhes.append({
                'material_id': str(estoque_item.material_id),
                'nome_categoria': estoque_item.nome_categoria,
                'quantidade_kg': str(estoque_item.quantidade_kg),
                'preco_sugerido': str(info_material.preco_sugerido) if info_material.preco_sugerido else None,
                'esta_disponivel': info_material.esta_disponivel,
            })

        if not detalhes:
            return {'status': 'error', 'message': 'Nenhum item de estoque bateu com os filtros informados.'}

        return {
            'status': 'ok',
            'quantidade_por_categoria_kg': {k: str(v) for k, v in quantidade_por_categoria.items()},
            'itens': detalhes,
        }

    def generalizar_estoque(
        self,
        config: RunnableConfig,
        material_ids: Optional[List[UUID]] = None,
        material_names: Optional[List[str]] = None,
        categoria_ids: Optional[List[UUID]] = None,
        disponivel: Optional[bool] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        preco_min: Optional[float] = None,
        preco_max: Optional[float] = None,
        incluir_sem_estoque: bool = False,
    ) -> dict:
        """Consulta o estoque disponível agrupado pelas categorias-pai (ex: Plástico, Metal,
        Papel, Vidro), em vez de por material/categoria específica. Use quando o usuário
        perguntar de forma genérica ('quanto tem de plástico no total') em vez de por um
        material específico ou solicitar um breve resumo do estoque atual."""
        print('[DEBUG]: acessou tool - generalizar_estoque')

        snapshots = config['configurable']['snapshots']
        estoque_snapshot: EstoqueSnapshot = snapshots['estoque_snapshot']
        material_snapshot: MaterialSnapshot = snapshots['material_snapshot']

        materiais_candidatos = self._resolver_materiais(material_snapshot, material_ids, material_names)
        if not materiais_candidatos:
            return {'status': 'error', 'message': 'Nenhum material correspondente encontrado.'}

        material_por_id = {item.material_id: item for item in materiais_candidatos}
        cat_ids_set = set(categoria_ids) if categoria_ids else None

        quantidade_por_categoria_pai: dict[str, Decimal] = defaultdict(Decimal)
        detalhes = []
        for estoque_item in estoque_snapshot.itens:
            info_material = material_por_id.get(estoque_item.material_id)
            if info_material is None:
                continue

            cat_pai_id, nome_cat_pai = self._resolver_categoria_pai_estoque(estoque_item)

            if cat_ids_set and not (estoque_item.categoria_id in cat_ids_set or cat_pai_id in cat_ids_set):
                continue

            if not self._passa_filtros(
                estoque_item, info_material,
                disponivel, quantidade_min_kg, quantidade_max_kg,
                preco_min, preco_max, incluir_sem_estoque,
            ):
                continue

            quantidade_por_categoria_pai[nome_cat_pai] += estoque_item.quantidade_kg
            detalhes.append({
                'material_id': str(estoque_item.material_id),
                'categoria_pai_id': str(cat_pai_id),
                'nome_categoria_pai': nome_cat_pai,
                'nome_categoria': estoque_item.nome_categoria,
                'quantidade_kg': str(estoque_item.quantidade_kg),
                'preco_sugerido': str(info_material.preco_sugerido) if info_material.preco_sugerido else None,
                'esta_disponivel': info_material.esta_disponivel,
            })

        if not detalhes:
            return {'status': 'error', 'message': 'Nenhum item de estoque bateu com os filtros informados.'}

        return {
            'status': 'ok',
            'quantidade_por_categoria_pai_kg': {k: str(v) for k, v in quantidade_por_categoria_pai.items()},
            'itens': detalhes,
        }

    def get_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.consultar_estoque,
                name='consultar_estoque',
                description=self.consultar_estoque.__doc__,
                args_schema=ConsultarEstoqueArgs,
            ),
            StructuredTool.from_function(
                func=self.generalizar_estoque,
                name='generalizar_estoque',
                description=self.generalizar_estoque.__doc__,
                args_schema=GeneralizarEstoqueArgs,
            ),
        ]