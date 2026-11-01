import unittest
from pathlib import Path
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app.main import app, _FRONTEND_DIR


_CHECKPOINTER_FAKE = object()


@asynccontextmanager
async def _abrir_checkpointer_fake(_settings):
    yield _CHECKPOINTER_FAKE


class FrontendServingTest(unittest.TestCase):
    def test_frontend_routes_served(self):
        container = SimpleNamespace(
            factories={},
            observabilidade=MagicMock(),
            memory_stores={},
        )
        with patch('app.main.Settings'), \
             patch('app.main.abrir_checkpointer', _abrir_checkpointer_fake), \
             patch('app.main.build_container', return_value=container):
            with TestClient(app) as client:
                # A raiz retorna o index.html, que referencia CSS e JS externos
                res_index = client.get('/')
                self.assertEqual(res_index.status_code, 200)
                self.assertIn('Renovai · Assistente', res_index.text)
                self.assertIn('chatForm', res_index.text)
                self.assertIn('./style.css', res_index.text)
                self.assertIn('./app.js', res_index.text)
                self.assertIn('data-tema', res_index.text)

                # Arquivos estáticos montados
                res_js = client.get('/app.js')
                self.assertEqual(res_js.status_code, 200)
                self.assertIn('CHAT_ENDPOINT', res_js.text)
                self.assertIn("/chat", res_js.text)

                res_css = client.get('/style.css')
                self.assertEqual(res_css.status_code, 200)
                self.assertIn('--wine:', res_css.text)
                self.assertIn('data-tema="EMPRESA"', res_css.text)

                # Assets do design system (logos por domínio e poses do Nimo)
                for asset in (
                    'logo_renovai_cooperativa.svg',
                    'logo_renovai_empresa.svg',
                    'cabeca_nimo.png',
                    'nimo_feliz.png',
                    'nimo_pensando.png',
                    'nimo_triste.png',
                ):
                    res_asset = client.get(f'/assets/{asset}')
                    self.assertEqual(res_asset.status_code, 200, asset)

                # Testa monitor.html
                res_mon = client.get('/monitor.html')
                self.assertEqual(res_mon.status_code, 200)


if __name__ == '__main__':
    unittest.main()
