from app.tools.base import Toolkit
from app.repository.postgresql.estoque_repository import EstoqueRepository, EstoqueSnapshot
from app.repository.postgresql.material_repository import MaterialRepository,  MaterialSnapshot

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
    preco_min: Optional[float] = Field(
        default=None,
        description="Preço sugerido mínimo do material, para perguntas sobre faixa de preço."
    )
    preco_max: Optional[float] = Field(
        default=None,
        description="Preço sugerido máximo do material, para perguntas sobre faixa de preço."
    )
    incluir_sem_estoque: bool = Field(
        default=False,
        description="Se true, inclui materiais com quantidade em estoque igual a zero no resultado (por padrão, ficam de fora)."
    )

class MaterialEstoqueToolkit(Toolkit):
    nome = 'material_estoque_tool'
    descricao = ''
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
    ):
        """Resolve a lista de materiais candidatos a partir de ids e/ou nomes (OR dentro do campo,
        igual à semântica combinada que definimos pro args_schema). Sem nenhum dos dois filtros,
        retorna todos os materiais do snapshot (equivalente a 'não filtrar por identidade')."""
        if not material_ids and not material_names:
            return list(material_snapshot.itens)

        ids_set = set(material_ids or [])
        termos = [
            self._MATERIAL_ALIASES.get(self._normalizar(n).upper(), self._normalizar(n))
            for n in (material_names or [])
        ]

        # NOTE: assumindo que o item de MaterialSnapshot expõe `.nome` (nome do material) — não
        # apareceu no arquivo original, que só usava `.material_id`, `.preco_sugerido` e
        # `.esta_disponivel`. Confirme o nome real do campo em material_repository.py.
        return [
            item for item in material_snapshot.itens
            if item.material_id in ids_set
            or any(termo in self._normalizar(item.nome) for termo in termos)
        ]

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


    def consultar_estoque (
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
        """Consulta o estoque disponível de um ou mais materiais, com filtros por disponibilidade,
        faixa de quantidade (kg) e faixa de preço sugerido."""

        print('[DEBUG]: acessou tool - consultar_estoque')

        snapshots = config["configurable"]["snapshots"]
        estoque_snapshot: EstoqueSnapshot = snapshots["estoque_snapshot"]
        material_snapshot: MaterialSnapshot = snapshots["material_snapshot"]

        materiais_candidatos = self._resolver_materiais(material_snapshot, material_ids, material_names)
        if not materiais_candidatos:
            return {"status": "error", "message": "Nenhum material correspondente encontrado."}

        material_por_id = {item.material_id: item for item in materiais_candidatos}

        quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
        detalhes = []
        for estoque_item in estoque_snapshot.itens:
            info_material = material_por_id.get(estoque_item.material_id)
            if info_material is None:
                continue  # não está entre os materiais filtrados por id/nome

            if not self._passa_filtros(
                estoque_item, info_material,
                disponivel, quantidade_min_kg, quantidade_max_kg,
                preco_min, preco_max, incluir_sem_estoque,
            ):
                continue

            quantidade_por_categoria[estoque_item.nome_categoria] += estoque_item.quantidade_kg
            detalhes.append({
                "material_id": str(estoque_item.material_id),
                "nome_categoria": estoque_item.nome_categoria,
                "quantidade_kg": str(estoque_item.quantidade_kg),
                "preco_sugerido": str(info_material.preco_sugerido) if info_material.preco_sugerido else None,
                "esta_disponivel": info_material.esta_disponivel,
            })

        if not detalhes:
            return {"status": "error", "message": "Nenhum item de estoque bateu com os filtros informados."}

        return {
            "status": "ok",
            "quantidade_por_categoria_kg": {k: str(v) for k, v in quantidade_por_categoria.items()},
            "itens": detalhes,
        }



    def get_tools(self):
        return [
            StructuredTool.from_function(
                func=self.consultar_estoque,
                name=self.nome,
                description=self.descricao,
                args_schema=ConsultarEstoqueArgs,
            ),
        ]