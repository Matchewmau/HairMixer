from django.test import TestCase


class APIDocsTests(TestCase):
    def test_schema_endpoint_available(self):
        r = self.client.get('/api/schema/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('openapi', r.json())

    def test_swagger_ui_served(self):
        r = self.client.get('/api/docs/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('text/html', r.headers.get('Content-Type', ''))

    def test_redoc_served(self):
        r = self.client.get('/api/redoc/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('text/html', r.headers.get('Content-Type', ''))
