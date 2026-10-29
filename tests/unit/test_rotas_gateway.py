"""O cálculo de rota responde em /ia/rotas/** (caminho que o gateway manda para a IA) e continua em
/rotas/** (chamadas internas do cluster)."""
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app


@asynccontextmanager
async def _abrir_checkpointer_fake(_settings):
    yield object()


def _resultado():
    return SimpleNamespace(
        origem='Praça da Sé, São Paulo',
        hora_saida=9,
        paradas_ordenadas=[],
        distancia_total_km=12.5,
        tempo_total_estimado_min=40.0,
        tempo_ordem_original_min=55.0,
        economia_estimada_min=15.0,
        economia_percentual=27.3,
    )


PEDIDO = {
    'cooperativa_id': 'coop-1',
    'rota_id': 'rota-1',
    'origem': {'id': 'e0', 'endereco': 'Praça da Sé, São Paulo'},
    'paradas': [{'id': 'e1', 'endereco': 'Avenida Paulista, 1000, São Paulo'}],
    'hora_saida': 9,
    'salvar_no_neo4j': False,
}


class RotasPeloGatewayTest(unittest.TestCase):
    def _cliente(self, modelo, repositorio):
        container = SimpleNamespace(
            factories={},
            modelo_rotas=modelo,
            repositories={'rota_repository': repositorio},
        )
        self._patches = [
            patch('app.main.Settings'),
            patch('app.main.abrir_checkpointer', _abrir_checkpointer_fake),
            patch('app.main.build_container', return_value=container),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])
        return TestClient(app)

    def test_otimizar_responde_em_ia_rotas_e_em_rotas(self):
        modelo, repositorio = MagicMock(), MagicMock()
        modelo.otimizar_rota.return_value = _resultado()
        with self._cliente(modelo, repositorio) as client:
            for caminho in ('/ia/rotas/otimizar', '/rotas/otimizar'):
                resposta = client.post(caminho, json=PEDIDO)
                self.assertEqual(resposta.status_code, 200, (caminho, resposta.text))
                corpo = resposta.json()
                self.assertEqual(corpo['tempo_ordem_original_min'], 55.0)
                self.assertEqual(corpo['economia_estimada_min'], 15.0)
                self.assertFalse(corpo['salvo_no_neo4j'])
        # salvar_no_neo4j=false: o app só pede o cálculo, o grafo é do RPA-neo4j
        repositorio.salvar_rota_otimizada.assert_not_called()
        self.assertEqual(modelo.otimizar_rota.call_count, 2)

    def test_retreinar_tambem_responde_em_ia_rotas(self):
        modelo, repositorio = MagicMock(), MagicMock()
        modelo.treinar_com_neo4j.return_value = {
            'mae_min': 1.5, 'r2_score': 0.9, 'amostras_neo4j': 10,
            'amostras_csv': 100, 'amostras_treino': 88, 'amostras_teste': 22,
        }
        with self._cliente(modelo, repositorio) as client:
            resposta = client.post('/ia/rotas/retreinar', json={})
        self.assertEqual(resposta.status_code, 200, resposta.text)
        self.assertEqual(resposta.json()['status'], 'ok')


if __name__ == '__main__':
    unittest.main()
