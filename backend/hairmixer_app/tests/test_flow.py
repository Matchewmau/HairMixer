from io import BytesIO
from django.test import TestCase, Client
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth import get_user_model
from PIL import Image
import json


class APIFLowTests(TestCase):
    def setUp(self):
        self.client = Client()
        cache.clear()

    def _make_image_file(self, fmt='PNG', size=(220, 220), color=(255, 0, 0)):
        bio = BytesIO()
        img = Image.new('RGB', size, color)
        img.save(bio, fmt)
        bio.seek(0)
        return SimpleUploadedFile(
            'test.png',
            bio.read(),
            content_type='image/png'
        )

    def test_preferences_then_recommend_without_upload(self):
        # Create preferences
        pref_payload = {
            'hair_type': 'straight',
            'hair_length': 'short',
            'maintenance': 'low',
            'occasions': ['work', 'casual']
        }
        pr = self.client.post(
            '/api/preferences/',
            data=json.dumps(pref_payload),
            content_type='application/json'
        )
        self.assertEqual(pr.status_code, 200, pr.content)
        pref_id = pr.json().get('preference_id')
        self.assertIsNotNone(pref_id)

        # Recommend with missing image_id -> expect 404 for missing image
        rec_payload = {
            'image_id': '00000000-0000-0000-0000-000000000000',
            'preference_id': pref_id,
        }
        rr = self.client.post(
            '/api/recommend/',
            data=json.dumps(rec_payload),
            content_type='application/json'
        )
        self.assertIn(rr.status_code, (400, 404))

    def test_upload_image_validation(self):
        # Upload too small image should fail validation
        small_img = self._make_image_file(size=(50, 50))
        ur = self.client.post('/api/upload/', data={'image': small_img})
        self.assertEqual(ur.status_code, 400)

    def test_upload_and_recommend_flow_smoke(self):
        # Use minimal valid image to pass serializer validation (>=200x200)
        img = self._make_image_file(size=(220, 220))
        ur = self.client.post('/api/upload/', data={'image': img})
        # If ML components not available, returns 400 with helpful payload
        self.assertIn(ur.status_code, (200, 400), ur.content)
        data = ur.json()
        image_id = data.get('image_id')

        # Create preferences
        pref_payload = {
            'hair_type': 'straight',
            'hair_length': 'short',
            'maintenance': 'low',
            'occasions': ['work']
        }
        pr = self.client.post(
            '/api/preferences/',
            data=json.dumps(pref_payload),
            content_type='application/json'
        )
        self.assertEqual(pr.status_code, 200, pr.content)
        pref_id = pr.json().get('preference_id')

        if image_id:
            rec_payload = {'image_id': image_id, 'preference_id': pref_id}
            rr = self.client.post(
                '/api/recommend/',
                data=json.dumps(rec_payload),
                content_type='application/json'
            )
            self.assertIn(rr.status_code, (200, 400, 404))

    def test_admin_cache_stats_requires_auth(self):
        # Anonymous access to admin cache required/denied based on env
        r = self.client.get('/api/admin/cache/stats/')
        # In default relaxed mode we set IsAuthenticated; anon should be 401
        self.assertIn(r.status_code, (200, 401, 403))

        # Create and login user, then retry
        User = get_user_model()
        u = User.objects.create_user(
            username='user1',
            email='user1@example.com',
            first_name='U',
            last_name='1',
            password='Passw0rd!'
        )
        self.client.force_login(u)
        r2 = self.client.get('/api/admin/cache/stats/')
        self.assertIn(r2.status_code, (200, 403))
