"""Filtra kwargs de tuning de latência pelos campos que a versão instalada da lib aceita."""
import logging

log = logging.getLogger(__name__)


def kwargs_suportados(cls, **kwargs) -> dict:
    campos = set(cls.model_fields) | {f.alias for f in cls.model_fields.values() if f.alias}
    aceitos = {k: v for k, v in kwargs.items() if k in campos}
    ignorados = sorted(set(kwargs) - set(aceitos))
    if ignorados:
        log.warning("%s: parâmetros ignorados (não suportados nesta versão): %s", cls.__name__, ignorados)
    return aceitos