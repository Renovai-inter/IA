from app.tools.base import Toolkit
from app.repository.postgresql.estoque_repository import EstoqueRepository, EstoqueSnapshot
from app.repository.postgresql.material_repository import MaterialRepository, MaterialSnapshot

from collections import defaultdict
from pydantic import BaseModel, Field
from decimal import Decimal
from uuid import UUID
from datetime import datetime
from typing import Optional, List

import unicodedata
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool


class ConsultarEstoqueArgs(BaseModel):
    material_ids: Optional[List[UUID]] = Field(
        default=None,
        description="IDs em materiais (FK). Pode conter um ou mais materiais."
    )
    material_nomes: Optional[List[str]] = Field(
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
    material_nomes: Optional[List[str]] = Field(
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

class BuscarEstoqueHistoricoArgs(BaseModel):
    data_inicio: Optional[datetime] = Field(None, description="Data de início do intervalo (ISO 8601)")
    data_fim: Optional[datetime] = Field(None, description="Data de fim do intervalo (ISO 8601)")
    material_ids: Optional[List[UUID]] = Field(
        default=None,
        description="IDs em materiais (FK). Pode conter um ou mais materiais."
    )
    material_nomes: Optional[List[str]] = Field(
        default=None,
        description="Nome (ou parte do nome) de um ou mais materiais, para busca quando o ID não é conhecido. Ex: ['PET', 'papelão']."
    )
    tipo_movimentacao: Optional[str] = Field(
        default=None,
        description=(
            "Tipo de movimentação (entrada ou saída) de um material no estoque."
            "'ENTRADA' = movimentação originada de uma triagem, quantidade_kg > 0"
            "'SAIDA' = movimentação originada de um pedido, quantidade < 0"
            "None = trazer TODAS as movimentações, entradas e saídas."
        )
    )
    disponivel: Optional[bool] = Field(default=None, description="Filtra por disponibilidade (true = apenas disponíveis).")
    quantidade_min_kg: Optional[float] = Field(default=None, description="Quantidade mínima em estoque (kg).")
    quantidade_max_kg: Optional[float] = Field(default=None, description="Quantidade máxima em estoque (kg).")
    preco_min: Optional[float] = Field(default=None, description="Preço sugerido mínimo do material.")
    preco_max: Optional[float] = Field(default=None, description="Preço sugerido máximo do material.")


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
        material_nomes: Optional[List[str]],
    ) -> list:
        """Resolve a lista de materiais candidatos a partir de ids e/ou nomes (OR dentro do campo).
        Sem nenhum dos dois filtros, retorna todos os materiais do snapshot."""
        if not material_ids and not material_nomes:
            return list(material_snapshot.itens)

        ids_set = set(material_ids or [])
        termos = [
            self._MATERIAL_ALIASES.get(self._normalizar(n).upper(), self._normalizar(n))
            for n in (material_nomes or [])
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

    def _passar_filtros(
        self,
        snapshot_item,
        info_material,
        disponivel: Optional[bool] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        preco_min: Optional[float] = None,
        preco_max: Optional[float] = None,
        incluir_sem_estoque: bool = False,
        # Filtro de histórico
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
        tipo_movimentacao: Optional[str] = None
    ) -> bool:
        if not incluir_sem_estoque and snapshot_item.quantidade_kg == 0:
            return False

        # quantidade_kg/preco_sugerido são Decimal, usando cast para evitar TypeError comparando com float
        if quantidade_min_kg is not None and abs(snapshot_item.quantidade_kg) < Decimal(str(quantidade_min_kg)):
            return False
        if quantidade_max_kg is not None and abs(snapshot_item.quantidade_kg) > Decimal(str(quantidade_max_kg)):
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

        if data_inicio is not None and snapshot_item.data_movimentacao < data_inicio:
            return False

        if data_fim is not None and snapshot_item.data_movimentacao > data_fim:
            return False

        if tipo_movimentacao is not None and snapshot_item.tipo_movimentacao.strip().upper() != tipo_movimentacao.strip().upper():
            return False

        return True


    def consultar_estoque(
        self,
        config: RunnableConfig,
        material_ids: Optional[List[UUID]] = None,
        material_nomes: Optional[List[str]] = None,
        disponivel: Optional[bool] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        preco_min: Optional[float] = None,
        preco_max: Optional[float] = None,
        incluir_sem_estoque: bool = False,
    ) -> dict:
        """Consulta e sumariza a quantidade atual de saldo em estoque de materiais.

        Filtra os registros com base em identificadores, categorias, disponibilidade,
        faixas de quantidade e valores sugeridos, utilizando os snapshots injetados
        na configuração. Retorna um resumo agregado das quantidades por categoria e a
        lista detalhada dos itens em estoque.

        Args:
            config (RunnableConfig): Configuração do LangChain/LangGraph contendo os
                snapshots de 'estoque_snapshot' e 'material_snapshot'.
            material_ids: Lista de UUIDs para filtrar materiais específicos.
            material_nomes (Optional[List[str]]): Lista de nomes ou termos de busca para materiais.
            disponivel (Optional[bool]): Status de disponibilidade do material.
            quantidade_min_kg: Quantidade mínima em kg para filtro.
            quantidade_max_kg: Quantidade máxima em kg para filtro.
            preco_min: Preço sugerido mínimo do material para filtro.
            preco_max: Preço sugerido máximo do material para filtro.
            incluir_sem_estoque (bool): Indica se deve incluir na busca itens cujo saldo 
                seja igual a zero. Padrão como False.

        Returns:
            dict: Dicionário contendo o status da operação ('ok' ou 'error') e:
                - Se sucesso: 'quantidade_por_categoria_kg' e a lista 'itens' com 
                  os detalhes de cada item consultado no estoque.
                - Se falha: 'message' descrevendo a razão do erro (ex: nenhum material 
                  ou saldo de estoque encontrado).
        """
        print('[DEBUG]: acessou tool - consultar_estoque')

        snapshots = config['configurable']['snapshots']
        estoque_snapshot: EstoqueSnapshot = snapshots['estoque_snapshot']
        material_snapshot: MaterialSnapshot = snapshots['material_snapshot']

        materiais_candidatos = self._resolver_materiais(material_snapshot, material_ids, material_nomes)
        if not materiais_candidatos:
            return {'status': 'error', 'message': 'Nenhum material correspondente encontrado.'}

        material_por_id = {item.material_id: item for item in materiais_candidatos}

        quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
        detalhes = []
        for estoque_item in estoque_snapshot.itens:
            info_material = material_por_id.get(estoque_item.material_id)
            if info_material is None:
                continue

            if not self._passar_filtros(
                snapshot_item=estoque_item,              info_material=info_material,
                quantidade_min_kg=quantidade_min_kg,     quantidade_max_kg=quantidade_max_kg,
                preco_min=preco_min,                     preco_max=preco_max,
                incluir_sem_estoque=incluir_sem_estoque, disponivel=disponivel,
            ):
                continue

            quantidade_por_categoria[estoque_item.nome_categoria] += estoque_item.quantidade_kg
            detalhes.append({
                'material_id':     str(estoque_item.material_id),
                'nome_categoria':  estoque_item.nome_categoria,
                'quantidade_kg':   str(estoque_item.quantidade_kg),
                'preco_sugerido':  str(info_material.preco_sugerido) if info_material.preco_sugerido else None,
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
        material_nomes: Optional[List[str]] = None,
        categoria_ids: Optional[List[UUID]] = None,
        disponivel: Optional[bool] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        preco_min: Optional[float] = None,
        preco_max: Optional[float] = None,
        incluir_sem_estoque: bool = False,
    ) -> dict:
        """Consulta e consolida o estoque agrupado por categorias-pai (macrocategorias).

        Agrupa os saldos de estoque por categorias genéricas (ex: Plástico, Metal, Papel,
        Vidro) em vez de subcategorias ou materiais específicos. Ideal para responder a
        perguntas abrangentes ou gerar visões executivas do volume total em estoque.

        Args:
            config (RunnableConfig): Configuração do LangChain/LangGraph contendo os
                snapshots de 'estoque_snapshot' e 'material_snapshot'.
            material_ids: Lista de UUIDs para filtrar materiais específicos.
            material_nomes (Optional[List[str]]): Lista de nomes ou termos de busca para materiais.
            categoria_ids: Lista de UUIDs para filtrar por categorias 
                ou categorias-pai específicas.
            disponivel (Optional[bool]): Status de disponibilidade do material.
            quantidade_min_kg: Quantidade mínima em kg para filtro.
            quantidade_max_kg: Quantidade máxima em kg para filtro.
            preco_min: Preço sugerido mínimo do material para filtro.
            preco_max: Preço sugerido máximo do material para filtro.
            incluir_sem_estoque (bool): Indica se deve incluir na busca itens cujo saldo 
                seja igual a zero. Padrão como False.

        Returns:
            dict: Dicionário contendo o status da operação ('ok' ou 'error') e:
                - Se sucesso: 'quantidade_por_categoria_pai_kg' com o total acumulado por 
                  macrocategoria e a lista 'itens' detalhada.
                - Se falha: 'message' descrevendo a razão do erro (ex: nenhum material 
                  ou saldo de estoque encontrado).
        """
        print('[DEBUG]: acessou tool - generalizar_estoque')

        snapshots = config['configurable']['snapshots']
        estoque_snapshot: EstoqueSnapshot = snapshots['estoque_snapshot']
        material_snapshot: MaterialSnapshot = snapshots['material_snapshot']

        materiais_candidatos = self._resolver_materiais(material_snapshot, material_ids, material_nomes)
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

            if not self._passar_filtros(
                snapshot_item=estoque_item,              info_material=info_material,
                quantidade_min_kg=quantidade_min_kg,     quantidade_max_kg=quantidade_max_kg,
                preco_min=preco_min,                     preco_max=preco_max,
                incluir_sem_estoque=incluir_sem_estoque, disponivel=disponivel,
            ):
                continue

            quantidade_por_categoria_pai[nome_cat_pai] += estoque_item.quantidade_kg
            detalhes.append({
                'material_id':        str(estoque_item.material_id),
                'categoria_pai_id':   str(cat_pai_id),
                'nome_categoria_pai': nome_cat_pai,
                'nome_categoria':     estoque_item.nome_categoria,
                'quantidade_kg':      str(estoque_item.quantidade_kg),
                'preco_sugerido':     str(info_material.preco_sugerido) if info_material.preco_sugerido else None,
                'esta_disponivel':    info_material.esta_disponivel,
            })

        if not detalhes:
            return {'status': 'error', 'message': 'Nenhum item de estoque bateu com os filtros informados.'}

        return {
            'status': 'ok',
            'quantidade_por_categoria_pai_kg': {k: str(v) for k, v in quantidade_por_categoria_pai.items()},
            'itens': detalhes,
        }

    def buscar_estoque_historico(
        self,
        config: RunnableConfig,
        data_inicio: Optional[datetime] = None,
        data_fim: Optional[datetime] = None,
        material_ids: Optional[List[UUID]] = None,
        material_nomes: Optional[List[str]] = None,
        tipo_movimentacao: Optional[str] = None,
        disponivel: Optional[bool] = None,
        quantidade_min_kg: Optional[float] = None,
        quantidade_max_kg: Optional[float] = None,
        preco_min: Optional[float] = None,
        preco_max: Optional[float] = None,
    ) -> dict:
        """Consulta e consolida o histórico de movimentações de estoque de materiais.

        Filtra os registros com base em critérios temporais, financeiros e operacionais
        utilizando os snapshots injetados na configuração. Retorna um resumo agregado por 
        categoria e por tipo de movimentação, além da lista detalhada dos itens encontrados.

        Args:
            config (RunnableConfig): Configuração do LangChain/LangGraph contendo os 
                snapshots de 'material_snapshot' e 'movimentacao_estoque_snapshot'.
            data_inicio: Data inicial para o filtro temporal de movimentação.
            data_fim: Data final para o filtro temporal de movimentação.
            material_ids: Lista de UUIDs para filtrar materiais específicos.
            material_nomes (Optional[List[str]]): Lista de nomes ou termos de busca para materiais.
            tipo_movimentacao (Optional[str]): Tipo da movimentação (ex: 'ENTRADA', 'SAIDA').
            disponivel (Optional[bool]): Status de disponibilidade do material.
            quantidade_min_kg: Quantidade mínima em kg para filtro.
            quantidade_max_kg: Quantidade máxima em kg para filtro.
            preco_min: Preço sugerido mínimo do material para filtro.
            preco_max: Preço sugerido máximo do material para filtro.

        Returns:
            dict: Dicionário contendo o status da operação ('ok' ou 'error') e:
                - Se sucesso: 'quantidade_por_categoria_kg', 'quantidade_por_tipo_movimentacao_kg'
                  e a lista 'itens' com os detalhes de cada movimentação.
                - Se falha: 'message' descrevendo a razão do erro (ex: nenhum material ou item encontrado).
        """
        print('[DEBUG]: acessou tool - buscar_estoque_historico')
        
        snapshots = config['configurable']['snapshots']
        material_snapshot = snapshots['material_snapshot']
        movimentacao_estoque_snapshot = snapshots['movimentacao_estoque_snapshot']
        
        materiais_candidatos = self._resolver_materiais(material_snapshot, material_ids, material_nomes)
        if not materiais_candidatos:
            return {'status': 'error', 'message': 'Nenhum material correspondente encontrado.'}

        material_por_id = {item.material_id: item for item in materiais_candidatos}

        quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
        quantidade_por_tipo_mov: dict[str, Decimal] = defaultdict(Decimal)
        detalhes = []
        for movimentacao_item in movimentacao_estoque_snapshot.itens:
            info_material = material_por_id.get(movimentacao_item.material_id)
            if info_material is None:
                continue

            if not self._passar_filtros(
                snapshot_item=movimentacao_item,     info_material=info_material,
                quantidade_min_kg=quantidade_min_kg, quantidade_max_kg=quantidade_max_kg,
                preco_min=preco_min,                 preco_max=preco_max,
                data_inicio=data_inicio,             data_fim=data_fim,
                tipo_movimentacao=tipo_movimentacao, disponivel=disponivel,
            ):
                continue

            quantidade_por_categoria[movimentacao_item.nome_categoria] += movimentacao_item.quantidade_kg
            quantidade_por_tipo_mov[movimentacao_item.tipo_movimentacao] += abs(movimentacao_item.quantidade_kg)

            detalhes.append({
                'material_id':       str(movimentacao_item.material_id),
                'nome_categoria':    movimentacao_item.nome_categoria,
                'quantidade_kg':     str(abs(movimentacao_item.quantidade_kg)),
                'tipo_movimentacao': movimentacao_item.tipo_movimentacao,
                'preco_sugerido':    str(info_material.preco_sugerido) if info_material.preco_sugerido else None,
                'esta_disponivel':   info_material.esta_disponivel,
                'data_movimentacao': str(movimentacao_item.data_movimentacao),
            })
        
        if not detalhes:
            return {'status': 'error', 'message': 'Nenhum item de estoque bateu com os filtros informados.'}
        
        return {
            'status': 'ok',
            'quantidade_por_categoria_kg': {k: str(v) for k, v in quantidade_por_categoria.items()},
            'quantidade_por_tipo_movimentacao_kg': {k: str(v) for k, v in quantidade_por_tipo_mov.items()},
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
            StructuredTool.from_function(
                func=self.buscar_estoque_historico,
                name='buscar_estoque_historico',
                description=self.buscar_estoque_historico.__doc__,
                args_schema=BuscarEstoqueHistoricoArgs,
            ),
        ]