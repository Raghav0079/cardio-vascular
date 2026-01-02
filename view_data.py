#!/usr/bin/env python3
"""
CardioPredict Pro - Data Viewer
View and analyze stored predictions from Supabase database
"""

import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class SupabaseDataViewer:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            print("❌ Missing Supabase credentials in .env file")
            self.connected = False
            return
        
        self.headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json'
        }
        
        self.connected = self.test_connection()
        
    def test_connection(self):
        """Test connection to Supabase"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=count",
                headers=self.headers,
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    def get_all_predictions(self):
        """Get all predictions from database"""
        if not self.connected:
            print("❌ Not connected to database")
            return []
        
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=*&order=timestamp.desc",
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {len(data)} predictions from database")
                return data
            else:
                print(f"❌ Error fetching data: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        if not self.connected:
            print("❌ Not connected to database")
            return []
        
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=*&order=timestamp.desc&limit={limit}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return []
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def display_summary(self):
        """Display summary statistics"""
        predictions = self.get_all_predictions()
        
        if not predictions:
            print("📊 No predictions found in database")
            return
        
        df = pd.DataFrame(predictions)
        
        print("\n" + "="*60)
        print("📊 CARDIOPREDICT PRO - DATABASE SUMMARY")
        print("="*60)
        
        # Basic statistics
        print(f"📈 Total Predictions: {len(df)}")
        print(f"📅 Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Risk level distribution
        if 'overall_result' in df.columns:
            print("\n🎯 Risk Level Distribution:")
            risk_counts = df['overall_result'].value_counts()
            for risk, count in risk_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {risk}: {count} ({percentage:.1f}%)")
        
        # Confidence levels
        if 'confidence_level' in df.columns:
            print("\n🔍 Confidence Levels:")
            conf_counts = df['confidence_level'].value_counts()
            for conf, count in conf_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {conf}: {count} ({percentage:.1f}%)")
        
        # Demographics
        if 'patient_age' in df.columns:
            print(f"\n👥 Patient Demographics:")
            print(f"   Average Age: {df['patient_age'].mean():.1f} years")
            print(f"   Age Range: {df['patient_age'].min()} - {df['patient_age'].max()}")
        
        if 'patient_sex' in df.columns:
            sex_counts = df['patient_sex'].value_counts()
            print(f"   Gender Distribution:")
            for sex, count in sex_counts.items():
                percentage = (count / len(df)) * 100
                print(f"     {sex}: {count} ({percentage:.1f}%)")
        
        print("="*60)
    
    def display_recent_predictions(self, limit=5):
        """Display recent predictions in a readable format"""
        predictions = self.get_recent_predictions(limit)
        
        if not predictions:
            print("📊 No recent predictions found")
            return
        
        print(f"\n📋 LAST {len(predictions)} PREDICTIONS")
        print("-" * 80)
        
        for i, pred in enumerate(predictions, 1):
            print(f"\n🏥 Prediction #{i}")
            print(f"   Patient: {pred.get('patient_name', 'Unknown')}")
            print(f"   Date: {pred.get('timestamp', '').split('T')[0]}")
            print(f"   Age: {pred.get('patient_age')} | Sex: {pred.get('patient_sex')}")
            print(f"   Result: {pred.get('overall_result')}")
            print(f"   Confidence: {pred.get('confidence_level')}")
            print(f"   Recommendation: {pred.get('recommendation', '')[:50]}...")
    
    def export_to_csv(self, filename=None):
        """Export all data to CSV file"""
        predictions = self.get_all_predictions()
        
        if not predictions:
            print("❌ No data to export")
            return
        
        if filename is None:
            filename = f"cardiopredict_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame(predictions)
        df.to_csv(filename, index=False)
        print(f"✅ Data exported to: {filename}")
        print(f"📊 Exported {len(df)} records")
    
    def search_by_risk_level(self, risk_level):
        """Search predictions by risk level"""
        if not self.connected:
            print("❌ Not connected to database")
            return []
        
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=*&overall_result=ilike.%{risk_level}%&order=timestamp.desc",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"🔍 Found {len(data)} predictions with '{risk_level}' risk")
                return data
            return []
                
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

def main():
    """Main function with interactive menu"""
    viewer = SupabaseDataViewer()
    
    if not viewer.connected:
        print("❌ Cannot connect to database. Check your .env file.")
        return
    
    print("🫀 CardioPredict Pro - Data Viewer")
    print("Connected to Supabase database ✅")
    
    while True:
        print("\n" + "="*40)
        print("SELECT AN OPTION:")
        print("1. 📊 View summary statistics")
        print("2. 📋 View recent predictions")
        print("3. 📁 Export all data to CSV")
        print("4. 🔍 Search by risk level")
        print("5. 🗄️ View all data (raw)")
        print("6. 🚪 Exit")
        print("="*40)
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            viewer.display_summary()
            
        elif choice == '2':
            limit = input("How many recent predictions? (default 5): ").strip()
            try:
                limit = int(limit) if limit else 5
            except:
                limit = 5
            viewer.display_recent_predictions(limit)
            
        elif choice == '3':
            filename = input("Enter filename (or press Enter for auto): ").strip()
            viewer.export_to_csv(filename if filename else None)
            
        elif choice == '4':
            risk = input("Enter risk level (High/Moderate/Low): ").strip()
            results = viewer.search_by_risk_level(risk)
            if results:
                df = pd.DataFrame(results)
                print(f"\n📊 Results for '{risk}' risk:")
                for i, pred in enumerate(results[:5], 1):
                    print(f"   {i}. {pred.get('patient_name')} - {pred.get('timestamp', '').split('T')[0]}")
            
        elif choice == '5':
            predictions = viewer.get_all_predictions()
            if predictions:
                df = pd.DataFrame(predictions)
                print(f"\n📊 ALL DATA ({len(predictions)} records):")
                print(df.to_string())
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()