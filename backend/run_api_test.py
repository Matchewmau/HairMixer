#!/usr/bin/env python3
"""
Simple script to test API endpoint
"""
import sys
import os
import time

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the test
import test_api_endpoint

if __name__ == '__main__':
    print("🔄 Starting API test...")
    time.sleep(2)  # Give server time to start
    
    # The test_api_endpoint module will run when imported
    print("✅ API test completed")
