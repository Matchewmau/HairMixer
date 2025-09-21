#!/usr/bin/env python3
"""
Simple script to run the API smoke test.
"""
import sys
import os
import time

# Ensure project root is on path so we can import sibling modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


def main():
    print("Starting API test...")
    # Give server a moment to be ready if just started
    time.sleep(1)
    try:
        from test_api_endpoint import test_api_endpoint
    except Exception as exc:
        print(f"❌ Failed to import test module: {exc}")
        return 1

    try:
        test_api_endpoint()
    except Exception as exc:
        print(f"Exception while running test: {exc}")
        return 2

    print("API test completed")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
