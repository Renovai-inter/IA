import csv
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
BAIRROS_CSV_PATH = DATA_DIR / 'sp_bairros_zonas.csv'
LOGRADOUROS_CSV_PATH = DATA_DIR / 'sp_logradouros_zonas.csv'


@dataclass(frozen=True)
class ZonaSP:
    codigo: int
    nome: str
    lat_ref: float
    lon_ref: float
    fator_retencao_viaria: float
    fator_tortuosidade: float


def _carregar_bairros_e_zonas(
    caminho_csv: Path = BAIRROS_CSV_PATH,
) -> tuple[dict[str, ZonaSP], dict[str, tuple[float, float, str]]]:
    zonas: dict[str, ZonaSP] = {}
    bairros: dict[str, tuple[float, float, str]] = {}

    with open(caminho_csv, mode='r', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            nome_bairro = linha['bairro'].strip().lower()
            sigla_zona = linha['zona'].strip().upper()
            cod_zona = int(linha['codigo_zona'])
            lat = float(linha['latitude'])
            lon = float(linha['longitude'])
            retencao = float(linha['fator_retencao_viaria'])
            tortuosidade = float(linha['fator_tortuosidade'])

            bairros[nome_bairro] = (lat, lon, sigla_zona)
            if sigla_zona not in zonas:
                zonas[sigla_zona] = ZonaSP(
                    codigo=cod_zona,
                    nome=sigla_zona,
                    lat_ref=lat,
                    lon_ref=lon,
                    fator_retencao_viaria=retencao,
                    fator_tortuosidade=tortuosidade,
                )

    return zonas, bairros


def _carregar_logradouros(
    caminho_csv: Path = LOGRADOUROS_CSV_PATH,
) -> dict[str, tuple[float, float, str, str]]:
    logradouros: dict[str, tuple[float, float, str, str]] = {}
    if not caminho_csv.exists():
        return logradouros

    with open(caminho_csv, mode='r', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        for linha in leitor:
            chave = linha['logradouro'].strip().lower()
            bairro = linha['bairro'].strip()
            sigla_zona = linha['zona'].strip().upper()
            lat = float(linha['latitude'])
            lon = float(linha['longitude'])
            logradouros[chave] = (lat, lon, sigla_zona, bairro)

    return logradouros


ZONAS_SP, BAIRROS_COORDS_SP = _carregar_bairros_e_zonas()
LOGRADOUROS_COORDS_SP = _carregar_logradouros()

_CEP_SP_RE = re.compile(r'\b(0[1-58]\d{3})-?(\d{3})\b')


def _normalizar_texto(texto: str) -> str:
    sem_acento = unicodedata.normalize('NFKD', texto or '').encode('ascii', 'ignore').decode('ascii')
    limpo = re.sub(r'\b(av|avenida)\.?\s+', 'av ', sem_acento.lower())
    limpo = re.sub(r'\b(r|rua)\.?\s+', 'rua ', limpo)
    limpo = re.sub(r'\b(pc|praca)\.?\s+', 'praca ', limpo)
    limpo = re.sub(r'\b(al|alameda)\.?\s+', 'alameda ', limpo)
    limpo = re.sub(r'\b(rod|rodovia)\.?\s+', 'rodovia ', limpo)
    limpo = re.sub(r'\b(estr|estrada)\.?\s+', 'estrada ', limpo)
    return re.sub(r'\s+', ' ', limpo).strip()


def classificar_zona_por_cep(cep: Optional[str]) -> str:
    if not cep:
        return 'CENTRO'
    m_cep = _CEP_SP_RE.search(str(cep))
    digitos = (m_cep.group(1) + m_cep.group(2)) if m_cep else re.sub(r'\D', '', str(cep))
    if len(digitos) < 5 or (not m_cep and len(digitos) != 8):
        return 'CENTRO'
    prefixo2 = int(digitos[:2])
    if prefixo2 == 1:
        return 'CENTRO'
    if prefixo2 == 2:
        return 'NORTE'
    if prefixo2 in (3, 8):
        return 'LESTE'
    if prefixo2 == 4:
        return 'SUL'
    if prefixo2 == 5:
        return 'OESTE'
    return 'CENTRO'


def inferir_coordenadas_sp(
    endereco: str = '',
    bairro: Optional[str] = None,
    cep: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> tuple[float, float, str]:
    """Resolve latitude, longitude e macro-zona de São Paulo a partir de coordenadas explícitas,
    catálogo de logradouros (`LOGRADOUROS_COORDS_SP`), catálogo de bairros em `data/sp_bairros_zonas.csv`
    ou faixa de CEP paulistano (`01000-000` a `08999-999`)."""
    texto_busca = _normalizar_texto(f'{bairro or ""} {endereco or ""}')

    for chave_log, (lat_l, lon_l, zona_l, _) in sorted(
        LOGRADOUROS_COORDS_SP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.search(rf'\b{re.escape(chave_log)}\b', texto_busca):
            lat = float(latitude) if latitude is not None else lat_l
            lon = float(longitude) if longitude is not None else lon_l
            return lat, lon, zona_l

    for nome_bairro, (lat_b, lon_b, zona_b) in sorted(
        BAIRROS_COORDS_SP.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.search(rf'\b{re.escape(nome_bairro)}\b', texto_busca):
            lat = float(latitude) if latitude is not None else lat_b
            lon = float(longitude) if longitude is not None else lon_b
            return lat, lon, zona_b

    m_cep_end = _CEP_SP_RE.search(endereco or '')
    cep_candidato = cep if cep else (m_cep_end.group(0) if m_cep_end else None)

    if cep_candidato:
        zona = classificar_zona_por_cep(cep_candidato)
        info_zona = ZONAS_SP[zona]
    elif latitude is not None and longitude is not None:
        zona = min(
            ZONAS_SP.keys(),
            key=lambda z: (ZONAS_SP[z].lat_ref - float(latitude)) ** 2 + (ZONAS_SP[z].lon_ref - float(longitude)) ** 2,
        )
        return float(latitude), float(longitude), zona
    else:
        nomes_zonas = ('CENTRO', 'NORTE', 'LESTE', 'SUL', 'OESTE')
        seed_zona = sum((idx + 3) * ord(ch) for idx, ch in enumerate(texto_busca or 'sp'))
        zona = nomes_zonas[seed_zona % len(nomes_zonas)]
        info_zona = ZONAS_SP[zona]

    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude), zona

    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(texto_busca or cep_candidato or 'sp'))
    delta_lat = ((seed % 100) - 50) * 0.0008
    delta_lon = (((seed // 7) % 100) - 50) * 0.0008
    return round(info_zona.lat_ref + delta_lat, 6), round(info_zona.lon_ref + delta_lon, 6), zona


def distancia_haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    raio_terra_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * raio_terra_km * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def fator_horario_sp(hora_saida: int) -> float:
    hora = int(hora_saida) % 24
    if 7 <= hora <= 9:
        return 1.55
    if 17 <= hora <= 19:
        return 1.65
    if 10 <= hora <= 16:
        return 1.22
    if 20 <= hora <= 21:
        return 1.15
    return 1.0


def extrair_features_trecho(
    lat_origem: float,
    lon_origem: float,
    zona_origem: str,
    lat_destino: float,
    lon_destino: float,
    zona_destino: str,
    hora_saida: int = 9,
    peso_estimado_kg: float = 200.0,
) -> list[float]:
    """Monta o vetor numérico de features para o estimador supervisionado scikit-learn:
        0: distancia_haversine_km
        1: distancia_viaria_estimada_km
        2: codigo_zona_origem
        3: codigo_zona_destino
        4: mudanca_de_zona
        5: fator_retencao_combinado
        6: fator_horario_sp
        7: hora_saida
        8: peso_estimado_kg
    """
    z_orig = ZONAS_SP.get(zona_origem, ZONAS_SP['CENTRO'])
    z_dest = ZONAS_SP.get(zona_destino, ZONAS_SP['CENTRO'])

    haversine = distancia_haversine_km(lat_origem, lon_origem, lat_destino, lon_destino)
    tortuosidade_media = (z_orig.fator_tortuosidade + z_dest.fator_tortuosidade) / 2.0
    mudanca_zona = 1.0 if z_orig.codigo != z_dest.codigo else 0.0
    if mudanca_zona > 0:
        tortuosidade_media += 0.08

    dist_viaria = haversine * tortuosidade_media
    retencao = (z_orig.fator_retencao_viaria + z_dest.fator_retencao_viaria) / 2.0
    mult_hora = fator_horario_sp(hora_saida)

    return [
        round(haversine, 4),
        round(dist_viaria, 4),
        float(z_orig.codigo),
        float(z_dest.codigo),
        mudanca_zona,
        round(retencao, 4),
        round(mult_hora, 4),
        float(int(hora_saida) % 24),
        float(max(0.0, peso_estimado_kg)),
    ]
