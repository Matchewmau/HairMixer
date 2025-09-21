from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from hairmixer_app.models import (
    HairstyleCategory,
    Hairstyle,
    UploadedImage,
    UserPreference,
)
import json


class DataDrivenEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.cat = HairstyleCategory.objects.create(name='Casual')

    def _mk_style(self, name, featured=False, pop=0.0, trend=0.0):
        return Hairstyle.objects.create(
            name=name,
            category=self.cat,
            is_active=True,
            is_featured=featured,
            popularity_score=pop,
            trend_score=trend,
        )

    def test_featured_returns_only_featured(self):
        self._mk_style('A', featured=True)
        self._mk_style('B', featured=True)
        self._mk_style('C', featured=False)

        r = self.client.get('/api/hairstyles/featured/?limit=10')
        self.assertEqual(r.status_code, 200)
        data = r.json()
        names = [x['name'] for x in data.get('featured_hairstyles', [])]
        self.assertEqual(set(names), {'A', 'B'})
        self.assertEqual(data.get('count'), 2)

    def test_trending_orders_by_popularity_when_no_signals(self):
        self._mk_style('Low', pop=1.0, trend=1.0)
        self._mk_style('Mid', pop=5.0, trend=2.0)
        self._mk_style('High', pop=10.0, trend=3.0)

        r = self.client.get('/api/hairstyles/trending/?limit=10')
        self.assertEqual(r.status_code, 200)
        data = r.json()
        names = [x['name'] for x in data.get('trending_hairstyles', [])]
        # Expect High first due to highest popularity_score
        self.assertGreaterEqual(len(names), 3)
        self.assertEqual(names[0], 'High')

    def test_search_text_fields(self):
        self._mk_style('Short Crop', featured=False)
        self._mk_style('Long Waves', featured=False)

        r = self.client.get('/api/search/', {'q': 'short'})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        names = [x['name'] for x in data.get('results', [])]
        self.assertIn('Short Crop', names)

    def test_recommendation_cache_hits_on_second_call(self):
        # Create minimal UploadedImage and UserPreference
        img_file = SimpleUploadedFile(
            'dummy.png', b'1234', content_type='image/png'
        )
        up = UploadedImage.objects.create(
            image=img_file,
            processing_status='completed',
            face_detected=True,
            face_count=1,
        )
        prefs = UserPreference.objects.create(
            hair_type='straight',
            hair_length='short',
            maintenance='low',
            occasions=['work'],
        )
        payload = {
            'image_id': str(up.id),
            'preference_id': str(prefs.id)
        }
        r1 = self.client.post(
            '/api/recommend/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(r1.status_code, 200)
        self.assertNotIn('from_cache', r1.json())

        r2 = self.client.post(
            '/api/recommend/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(r2.status_code, 200)
        self.assertTrue(r2.json().get('from_cache', False))
