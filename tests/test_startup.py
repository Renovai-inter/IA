import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from app.main import app


class StartupTest(unittest.TestCase):
    def test_health_and_cleanup_after_connection_error(self):
        failing, healthy = MagicMock(), MagicMock()
        failing.close.side_effect = RuntimeError('connection already closed')
        container = SimpleNamespace(factories={
            'first': (failing, 'first-connection'),
            'second': (healthy, 'second-connection'),
        })
        with patch('app.main.Settings'), \
             patch('app.main.build_container', return_value=container), \
             self.assertLogs('fastapi', level='ERROR'):
            with TestClient(app) as client:
                response = client.get('/health')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {'status': 'ok'})
                self.assertIs(app.state.container, container)
        healthy.close.assert_called_once_with('second-connection')


if __name__ == '__main__':
    unittest.main()
