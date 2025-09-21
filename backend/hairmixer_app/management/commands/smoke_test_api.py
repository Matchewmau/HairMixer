from django.core.management.base import BaseCommand
import json
import sys
import time
try:
    import requests
except Exception:
    requests = None


class Command(BaseCommand):
    help = "Run a lightweight API smoke test against the local server"

    def add_arguments(self, parser):
        parser.add_argument(
            '--base-url',
            default='http://127.0.0.1:8000',
            help='Base URL of the server to test',
        )
        parser.add_argument(
            '--delay',
            type=float,
            default=0.0,
            help='Delay seconds before firing requests',
        )

    def handle(self, *args, **options):
        base = options['base_url'].rstrip('/')
        if requests is None:
            self.stderr.write(
                self.style.ERROR('requests not installed in this environment')
            )
            sys.exit(2)

        if options['delay']:
            time.sleep(options['delay'])

        paths = [
            ('health', '/api/health/'),
            ('featured', '/api/hairstyles/featured/'),
            ('trending', '/api/hairstyles/trending/'),
            ('search', '/api/search/?q=layer'),
        ]

        ok = True
        summary = {}
        for name, path in paths:
            url = f"{base}{path}"
            try:
                r = requests.get(url, timeout=10)
                passed = r.status_code in (200, 400)
                # Record small body snippet
                body = r.text[:200]
                summary[name] = {
                    'status': r.status_code,
                    'ok': passed,
                    'snippet': body,
                }
                if not passed:
                    ok = False
                self.stdout.write(
                    f"[{name}] {r.status_code} -> "
                    f"{('OK' if passed else 'FAIL')} : {url}"
                )
            except Exception as exc:
                ok = False
                summary[name] = {'error': str(exc), 'ok': False}
                self.stderr.write(self.style.ERROR(f"[{name}] ERROR: {exc}"))

        self.stdout.write(json.dumps(summary, indent=2))
        if not ok:
            self.stderr.write(
                self.style.ERROR('Smoke test encountered failures')
            )
            sys.exit(1)
        self.stdout.write(self.style.SUCCESS('Smoke test passed'))
