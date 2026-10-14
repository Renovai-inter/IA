import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app.main import app


_CHECKPOINTER_FAKE = object()


@asynccontextmanager
async def _abrir_checkpointer_fake(_settings):
    # Settings é um MagicMock neste teste: REDIS_URL viraria um valor "truthy"
    # e o lifespan tentaria conectar num Redis de verdade.
    yield _CHECKPOINTER_FAKE


class StartupTest(unittest.TestCase):
    def test_health_and_cleanup_after_connection_error(self):
        failing, healthy = MagicMock(), MagicMock()
        failing.close.side_effect = RuntimeError('connection already closed')
        container = SimpleNamespace(factories={
            'first': (failing, 'first-connection'),
            'second': (healthy, 'second-connection'),
        })
        with patch('app.main.Settings'), \
             patch('app.main.abrir_checkpointer', _abrir_checkpointer_fake), \
             patch('app.main.build_container', return_value=container) as build, \
             self.assertLogs('fastapi', level='ERROR'):
            with TestClient(app) as client:
                response = client.get('/health')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'status': 'ok'})
                self.assertIs(app.state.container, container)
        healthy.close.assert_called_once_with('second-connection')
        # o checkpointer aberto no lifespan é o que chega no container
        self.assertIs(build.call_args.kwargs['checkpointer'], _CHECKPOINTER_FAKE)


if __name__ == '__main__':
    unittest.main()
