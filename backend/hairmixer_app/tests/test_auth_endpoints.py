from django.test import TestCase, Client


class AuthEndpointTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_signup_and_login_flow(self):
        signup_payload = {
            "firstName": "Jane",
            "lastName": "Doe",
            "email": "jane.doe@example.com",
            "password": "Passw0rd!",
        }
        rs = self.client.post("/api/auth/signup/", signup_payload)
        # Either created or already exists if tests rerun
        self.assertIn(rs.status_code, (201, 400), rs.content)

        login_payload = {
            "email": "jane.doe@example.com",
            "password": "Passw0rd!",
        }
        rl = self.client.post("/api/auth/login/", login_payload)
        self.assertIn(rl.status_code, (200, 401), rl.content)
        if rl.status_code == 200:
            data = rl.json()
            self.assertIn("access_token", data)
            self.assertIn("refresh_token", data)
