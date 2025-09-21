from django.test import TestCase, Client


class APISmokeTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_health_endpoint(self):
        resp = self.client.get('/api/health/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get('status'), 'ok')
        self.assertIn('ml_available', data)

    def test_featured_endpoint(self):
        resp = self.client.get('/api/hairstyles/featured/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('featured_hairstyles', data)
        self.assertIsInstance(data['featured_hairstyles'], list)

    def test_trending_endpoint(self):
        resp = self.client.get('/api/hairstyles/trending/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('trending_hairstyles', data)
        self.assertIsInstance(data['trending_hairstyles'], list)

    def test_search_endpoint(self):
        resp = self.client.get('/api/search/', {'q': 'short'})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('results', data)
        self.assertIsInstance(data['results'], list)
        self.assertIn('pagination', data)
