#!/usr/bin/env python3
"""
Database Viewer for CardioPredict Pro
View and analyze stored predictions from Supabase
"""

import os
import requests
import json
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatabaseViewer:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            print("❌ Missing Supabase credentials in .env file")
            return
            
        self.headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        
        print("✅ Connected to database")
    
    def get_all_predictions(self, limit=50):
        """Get all predictions from database"""
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=*&order=timestamp.desc&limit={limit}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Database error: {e}")
            return []
    
    def get_statistics(self):
        """Get database statistics"""
        try:
            # Total count
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=*",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                total_predictions = len(data)
                
                # Analyze results
                high_risk = len([p for p in data if 'High Risk' in p.get('overall_result', '')])
                moderate_risk = len([p for p in data if 'Moderate Risk' in p.get('overall_result', '')])
                low_risk = len([p for p in data if 'Low Risk' in p.get('overall_result', '')])
                
                # Recent predictions (last 24 hours)
                recent = [p for p in data if self._is_recent(p.get('timestamp', ''))]
                
                return {
                    'total_predictions': total_predictions,
                    'high_risk': high_risk,
                    'moderate_risk': moderate_risk,
                    'low_risk': low_risk,
                    'recent_24h': len(recent)
                }
            else:
                return {}
                
        except Exception as e:
            print(f"❌ Statistics error: {e}")
            return {}
    
    def _is_recent(self, timestamp_str):
        """Check if timestamp is within last 24 hours"""
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now().astimezone()
            diff = now - timestamp.replace(tzinfo=now.tzinfo)
            return diff.total_seconds() < 86400  # 24 hours
        except:
            return False
    
    def print_summary(self):
        """Print database summary"""
        print("\n" + "="*60)
        print("📊 CARDIOPREDICT PRO - DATABASE ANALYTICS")
        print("="*60)
        
        stats = self.get_statistics()
        if not stats:
            print("❌ Unable to fetch statistics")
            return
        
        print(f"📈 Total Predictions: {stats['total_predictions']}")
        print(f"🔴 High Risk Patients: {stats['high_risk']}")
        print(f"🟡 Moderate Risk Patients: {stats['moderate_risk']}")
        print(f"🟢 Low Risk Patients: {stats['low_risk']}")
        print(f"⏰ Recent (24h): {stats['recent_24h']}")
        
        # Risk distribution
        total = stats['total_predictions']
        if total > 0:
            print(f"\n📊 Risk Distribution:")
            print(f"   High Risk: {stats['high_risk']/total*100:.1f}%")
            print(f"   Moderate Risk: {stats['moderate_risk']/total*100:.1f}%")
            print(f"   Low Risk: {stats['low_risk']/total*100:.1f}%")
    
    def print_recent_predictions(self, limit=10):
        """Print recent predictions"""
        print(f"\n📋 RECENT PREDICTIONS (Last {limit})")
        print("-" * 60)
        
        predictions = self.get_all_predictions(limit)
        if not predictions:
            print("❌ No predictions found")
            return
        
        for i, pred in enumerate(predictions, 1):
            timestamp = pred.get('timestamp', '')[:19].replace('T', ' ')
            patient = pred.get('patient_name', 'Anonymous')
            result = pred.get('overall_result', 'Unknown')
            confidence = pred.get('confidence_level', 'Unknown')
            
            print(f"{i:2}. {timestamp} | {patient[:15]:15} | {result:20} | {confidence}")
    
    def export_to_csv(self, filename="predictions_export.csv"):
        """Export all predictions to CSV"""
        predictions = self.get_all_predictions(1000)  # Get more for export
        if not predictions:
            print("❌ No data to export")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        df.to_csv(filename, index=False)
        print(f"✅ Exported {len(predictions)} predictions to {filename}")

def main():
    """Main function"""
    viewer = DatabaseViewer()
    
    if not hasattr(viewer, 'headers'):
        return
    
    while True:
        print("\n" + "="*40)
        print("CARDIOPREDICT PRO - DATABASE VIEWER")
        print("="*40)
        print("1. 📊 Show Statistics")
        print("2. 📋 Show Recent Predictions") 
        print("3. 💾 Export to CSV")
        print("4. 🔄 Refresh Data")
        print("5. ❌ Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            viewer.print_summary()
        elif choice == '2':
            limit = input("How many recent predictions? (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            viewer.print_recent_predictions(limit)
        elif choice == '3':
            filename = input("Export filename (default: predictions_export.csv): ").strip()
            filename = filename or "predictions_export.csv"
            viewer.export_to_csv(filename)
        elif choice == '4':
            print("🔄 Refreshing...")
        elif choice == '5':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()