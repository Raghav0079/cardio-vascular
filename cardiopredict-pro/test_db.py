#!/usr/bin/env python3
"""Test database connection and check for data"""

from database_integration import SupabaseDB
import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

print('🔍 Checking database connection...')
print(f'SUPABASE_URL: {os.getenv("SUPABASE_URL", "Not found")}')
supabase_key = os.getenv("SUPABASE_ANON_KEY", "Not found")
print(f'SUPABASE_ANON_KEY: {supabase_key[:20] if len(supabase_key) > 20 else "Not found"}...')

# Test database connection
print('\n🔌 Testing database connection...')
db = SupabaseDB()
print(f'Database connected: {db.connected}')

if db.connected:
    print('\n📊 Testing database queries...')
    
    # Test 1: Check if predictions table exists
    try:
        response = requests.get(
            f'{db.supabase_url}/rest/v1/predictions?select=count',
            headers=db.headers,
            timeout=10
        )
        print(f'Table query status: {response.status_code}')
        if response.status_code == 200:
            print('✅ Predictions table exists')
            data = response.json()
            print(f'Table response: {data}')
        else:
            print(f'❌ Table query failed: {response.text}')
    except Exception as e:
        print(f'❌ Table query error: {e}')
    
    # Test 2: Try to get recent predictions
    print('\n📋 Testing get_recent_predictions...')
    recent = db.get_recent_predictions(5)
    print(f'Recent predictions: {len(recent) if recent else 0} records')
    if recent:
        print('First record keys:', list(recent[0].keys()) if recent[0] else 'No data')
    
    # Test 3: Check if table is empty or needs to be created
    print('\n🏗️ Checking if table needs to be created...')
    try:
        # Try a simple select to see table structure
        response = requests.get(
            f'{db.supabase_url}/rest/v1/predictions?limit=1',
            headers=db.headers,
            timeout=10
        )
        print(f'Table structure check: {response.status_code}')
        if response.status_code == 200:
            data = response.json()
            print(f'Table data: {data}')
            if not data:
                print('📭 Table exists but is empty (no predictions made yet)')
        elif response.status_code == 406:
            print('❌ Table does not exist - needs to be created')
        else:
            print(f'❌ Unknown error: {response.text}')
    except Exception as e:
        print(f'❌ Structure check error: {e}')

else:
    print('❌ Database connection failed')
    print('Please check:')
    print('1. Supabase URL is correct')
    print('2. Supabase anon key is valid') 
    print('3. Internet connection is working')