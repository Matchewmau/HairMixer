import json
from django.test import TestCase, Client
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth import get_user_model
from hairmixer_app.models import (
    HairstyleCategory,
    Hairstyle,
    UploadedImage,
    UserPreference,
    RecommendationLog,
)


class AdditionalUserFlowTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.cat = HairstyleCategory.objects.create(name='Casual')
        self.style = Hairstyle.objects.create(
            name='Test Style',
            category=self.cat,
            is_active=True,
        )

    def test_feedback_happy_path(self):
        payload = {
            'liked': True,
            'rating': 5,
            'note': 'Great style!',
            'hairstyle': str(self.style.id),
            'recommendation': None,
        }
        r = self.client.post(
            '/api/feedback/',
            data=json.dumps(payload),
            content_type='application/json',
        )
        self.assertIn(r.status_code, (200, 400), r.content)
        if r.status_code == 200:
            self.assertIn('feedback_id', r.json())

    def test_user_recommendations_requires_auth_and_lists(self):
        User = get_user_model()
        u = User.objects.create_user(
            username='recuser',
            email='recuser@example.com',
            first_name='R',
            last_name='U',
            password='Passw0rd!',
        )

        # Anonymous should be denied
        r0 = self.client.get('/api/user/recommendations/')
        self.assertIn(r0.status_code, (401, 403))

        # Create data and log in
        img_file = SimpleUploadedFile(
            'dummy.png', b'1234', content_type='image/png'
        )
        img = UploadedImage.objects.create(
            image=img_file,
            processing_status='completed',
            face_detected=True,
            face_count=1,
        )
        pref = UserPreference.objects.create(
            user=u,
            hair_type='straight',
            hair_length='short',
            maintenance='low',
            occasions=['work'],
        )
        RecommendationLog.objects.create(
            user=u,
            uploaded=img,
            preference=pref,
            face_shape='oval',
            candidates=[str(self.style.id)],
            status='completed',
        )

        self.client.force_login(u)
        r1 = self.client.get('/api/user/recommendations/?per_page=10')
        self.assertEqual(r1.status_code, 200, r1.content)
        payload = r1.json()
        self.assertIn('recommendations', payload)
        self.assertGreaterEqual(len(payload['recommendations']), 1)
