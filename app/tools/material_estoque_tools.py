from decimal import Decimal
import unicodedata
from collections import defaultdict

from langchain_core.runnables import RunnableConfig
from langchain.tools import tool

from app.repository.postgresql.estoque_repository import EstoqueSnapshot
from app.repository.postgresql.material_repository import MaterialSnapshot


def _normalizar(texto: str) -> str:
    """Remove acentos, underscore/hífen e baixa a caixa, pra comparação tolerante."""
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")
    return sem_acento.lower().replace("_", " ").replace("-", " ").strip()


@tool
def consultar_estoque(material: str, config: RunnableConfig) -> dict:
    """Consulta o estoque disponível de um material ou categoria.

    Args:
        material: termo de busca livre, em português natural — ex: "papel",
            "plástico", "pet". Não normalize para snake_case nem remova acentos;
            passe como o usuário mencionou.
    """
    estoque_snapshot: EstoqueSnapshot = config["configurable"]["snapshots"]["estoque_snapshot"]
    material_snapshot: MaterialSnapshot = config["configurable"]["snapshots"]["material_snapshot"]

    material_por_id = {item.material_id: item for item in material_snapshot.itens}
    termo = _normalizar(material)

    encontrados = [
        estoque_item for estoque_item in estoque_snapshot.itens
        if termo in _normalizar(estoque_item.nome_categoria)
    ]

    if not encontrados:
        return {
            "status": "error",
            "message": f"Nenhuma categoria correspondente a '{material}' encontrada no estoque.",
        }

    quantidade_por_categoria: dict[str, Decimal] = defaultdict(Decimal)
    detalhes = []
    for estoque_item in encontrados:
        info_material = material_por_id.get(estoque_item.material_id)
        quantidade_por_categoria[estoque_item.nome_categoria] += estoque_item.quantidade_kg
        detalhes.append({
            "material_id": str(estoque_item.material_id),
            "nome_categoria": estoque_item.nome_categoria,
            "quantidade_kg": str(estoque_item.quantidade_kg),
            "preco_sugerido": str(info_material.preco_sugerido) if info_material and info_material.preco_sugerido else None,
            "esta_disponivel": info_material.esta_disponivel if info_material else None,
        })

    return {
        "status": "ok",
        "quantidade_por_categoria_kg": {k: str(v) for k, v in quantidade_por_categoria.items()},
        "itens": detalhes,
    }


MATERIAL_ESTOQUE_TOOLS = [consultar_estoque]