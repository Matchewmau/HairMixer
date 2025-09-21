from django.test import TestCase


class OverlayDocsTests(TestCase):
    def test_overlay_in_schema_paths(self):
        r = self.client.get('/api/schema/')
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # ensure POST /api/overlay/ documented
        paths = data.get('paths', {})
        self.assertIn('/api/overlay/', paths)
        self.assertIn('post', paths['/api/overlay/'])
