#!/usr/bin/env python3
"""
Automatically create the predictions table in Supabase database
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_database_table():
    """Create the predictions table using Supabase REST API"""
    
    supabase_url = os.getenv('SUPABASE_URL')
    anon_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not anon_key:
        print("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY environment variables")
        return False
    
    print("🏗️ Setting up database table...")
    print(f"🔗 Connecting to: {supabase_url}")
    
    # SQL to create the table
    create_table_sql = """
    -- Enable UUID extension
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    
    -- Create predictions table
    CREATE TABLE IF NOT EXISTS public.predictions (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        
        -- Patient Information
        patient_name TEXT NOT NULL,
        patient_age INTEGER NOT NULL,
        patient_sex TEXT NOT NULL,
        
        -- Clinical Parameters  
        chest_pain_type TEXT NOT NULL,
        resting_bp INTEGER NOT NULL,
        cholesterol INTEGER NOT NULL,
        fasting_blood_sugar TEXT NOT NULL,
        rest_ecg TEXT NOT NULL,
        max_heart_rate INTEGER NOT NULL,
        exercise_angina TEXT NOT NULL,
        st_depression DECIMAL(4,2) NOT NULL,
        slope TEXT NOT NULL,
        colored_vessels INTEGER NOT NULL,
        thalassemia TEXT NOT NULL,
        
        -- AI Prediction Results
        positive_predictions INTEGER NOT NULL,
        confidence_level TEXT NOT NULL,
        overall_result TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        model_results JSONB NOT NULL,
        model_probabilities JSONB NOT NULL
    );
    
    -- Enable Row Level Security
    ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;
    
    -- Create policies for anonymous access
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_policies 
            WHERE tablename = 'predictions' AND policyname = 'Enable read access for all users'
        ) THEN
            CREATE POLICY "Enable read access for all users" ON public.predictions
                FOR SELECT USING (true);
        END IF;
        
        IF NOT EXISTS (
            SELECT 1 FROM pg_policies 
            WHERE tablename = 'predictions' AND policyname = 'Enable insert access for all users'  
        ) THEN
            CREATE POLICY "Enable insert access for all users" ON public.predictions
                FOR INSERT WITH CHECK (true);
        END IF;
    END
    $$;
    
    -- Grant permissions
    GRANT USAGE ON SCHEMA public TO anon;
    GRANT SELECT, INSERT ON public.predictions TO anon;
    """
    
    # Execute SQL via Supabase RPC
    headers = {
        'apikey': anon_key,
        'Authorization': f'Bearer {anon_key}',
        'Content-Type': 'application/json',
        'Prefer': 'return=minimal'
    }
    
    try:
        # Use Supabase RPC to execute SQL
        response = requests.post(
            f'{supabase_url}/rest/v1/rpc/execute_sql',
            headers=headers,
            json={'query': create_table_sql},
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Database table created successfully!")
            return True
        else:
            print(f"❌ Failed to create table: {response.status_code}")
            print(f"Response: {response.text}")
            print("\n📋 Manual Setup Required:")
            print("1. Go to your Supabase Dashboard: https://viytfimxtxwwbslygide.supabase.co")
            print("2. Click on 'SQL Editor'")
            print("3. Copy and paste the SQL from 'database_schema.sql'")
            print("4. Run the SQL query to create the table")
            return False
            
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        print("\n📋 Manual Setup Required:")
        print("1. Go to your Supabase Dashboard: https://viytfimxtxwwbslygide.supabase.co")
        print("2. Click on 'SQL Editor'") 
        print("3. Copy and paste the SQL from 'database_schema.sql'")
        print("4. Run the SQL query to create the table")
        return False

def test_table_creation():
    """Test if the table was created successfully"""
    from database_integration import SupabaseDB
    
    print("\n🔍 Testing table creation...")
    db = SupabaseDB()
    
    if not db.connected:
        print("❌ Database connection failed")
        return False
    
    try:
        response = requests.get(
            f'{db.supabase_url}/rest/v1/predictions?limit=1',
            headers=db.headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Table created and accessible!")
            print("🎉 Your database is ready for storing predictions!")
            return True
        else:
            print(f"❌ Table test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Table test error: {e}")
        return False

if __name__ == "__main__":
    print("🗄️ CardioPredict Pro Database Setup")
    print("=" * 40)
    
    success = create_database_table()
    if success:
        test_table_creation()
    else:
        print("\n⚠️ Automatic setup failed. Please set up manually using Supabase Dashboard.")