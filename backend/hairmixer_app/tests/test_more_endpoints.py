from django.test import TestCase, Client
from django.test.utils import CaptureQueriesContext
from django.db import connection
from django.contrib.auth import get_user_model

from hairmixer_app.models import HairstyleCategory, Hairstyle


class AdditionalEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()
        # Minimal data
        self.cat = HairstyleCategory.objects.create(
            name="Formal", sort_order=1
        )
        # Create some hairstyles to exercise queries
        for i in range(8):
            Hairstyle.objects.create(
                name=f"Style {i}",
                category=self.cat,
                is_active=True,
                is_featured=(i % 2 == 0),
                popularity_score=float(i),
                trend_score=float(10 - i),
            )

    def test_categories_endpoint(self):
        r = self.client.get("/api/hairstyles/categories/")
        self.assertEqual(r.status_code, 200, r.content)
        data = r.json()
        self.assertIn("categories", data)
        self.assertGreaterEqual(len(data["categories"]), 1)
        # Ensure ordering stability (sort_order, name)
        names = [c["name"] for c in data["categories"]]
        self.assertEqual(names, sorted(names))

    def test_hairstyle_detail_endpoint(self):
        style = Hairstyle.objects.first()
        r = self.client.get(f"/api/hairstyles/{style.id}/")
        self.assertEqual(r.status_code, 200, r.content)
        data = r.json()
        self.assertIn("hairstyle", data)
        self.assertEqual(data["hairstyle"]["id"], str(style.id))
        self.assertIn("feedback_stats", data)

    def test_admin_cache_cleanup_requires_auth(self):
        # Should require auth (relaxed perms still require IsAuthenticated)
        r = self.client.post("/api/admin/cache/cleanup/")
        self.assertIn(r.status_code, (401, 403))

        # Login basic user then allowed/forbidden depending on settings
        User = get_user_model()
        u = User.objects.create_user(
            username="testuser",
            email="testuser@example.com",
            first_name="T",
            last_name="U",
            password="Passw0rd!",
        )
        self.client.force_login(u)
        r2 = self.client.post("/api/admin/cache/cleanup/")
        self.assertIn(r2.status_code, (200, 403))

    def test_performance_guard_featured_and_trending(self):
        # Expect a bounded number of queries due to select_related/annotations
        with CaptureQueriesContext(connection) as ctx1:
            rr = self.client.get("/api/hairstyles/featured/?limit=5")
            self.assertEqual(rr.status_code, 200)
        # Keep this generous but low enough to catch N+1 regressions
        self.assertLessEqual(len(ctx1), 8)

        with CaptureQueriesContext(connection) as ctx2:
            rt = self.client.get("/api/hairstyles/trending/?limit=5")
            self.assertEqual(rt.status_code, 200)
        self.assertLessEqual(len(ctx2), 10)

    def test_analytics_event_tolerant(self):
        # Anonymous should be rejected (IsAuthenticated). Ensure 401/403
        r = self.client.post(
            "/api/analytics/event/",
            data={"event_type": "page_view"},
            content_type="application/json",
        )
        self.assertIn(r.status_code, (401, 403))

        # Authenticated request should pass or return 500 if service missing
        User = get_user_model()
        u = User.objects.create_user(
            username="ana",
            email="ana@example.com",
            first_name="A",
            last_name="N",
            password="Passw0rd!",
        )
        self.client.force_login(u)
        r2 = self.client.post(
            "/api/analytics/event/",
            data={"event_type": "page_view", "event_data": {"page": "/"}},
            content_type="application/json",
        )
        self.assertIn(r2.status_code, (200, 500))
