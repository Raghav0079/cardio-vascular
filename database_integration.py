"""
Supabase Database Integration for CardioPredict Pro
Records all predictions and patient data for analytics and tracking
"""

import os
import json
import requests
from datetime import datetime
import uuid

class SupabaseDB:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.headers = {
            'apikey': self.supabase_key,
            'Authorization': f'Bearer {self.supabase_key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }
        self.connected = self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        if not self.supabase_url or not self.supabase_key:
            return False
        
        try:
            # Test with a simple query
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=id&limit=1",
                headers=self.headers,
                timeout=5
            )
            return response.status_code in [200, 404]  # 404 means table doesn't exist yet, but connection works
        except:
            return False
    
    def create_tables(self):
        """Create necessary tables if they don't exist"""
        if not self.connected:
            return False
        
        # This would typically be done via Supabase dashboard SQL editor
        # Tables: predictions, patients, model_results
        return True
    
    def save_prediction(self, patient_data, prediction_results, model_outputs):
        """Save complete prediction record to database"""
        if not self.connected:
            print("⚠️ Database not connected - skipping save")
            return None
        
        try:
            # Generate unique session ID
            session_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Prepare prediction record
            prediction_record = {
                'id': session_id,
                'timestamp': timestamp,
                'patient_name': patient_data.get('patient_name', 'Anonymous'),
                'patient_age': patient_data.get('age', 0),
                'patient_sex': patient_data.get('sex', 'Unknown'),
                'chest_pain_type': patient_data.get('chest_pain_type', ''),
                'resting_bp': patient_data.get('resting_bp', 0),
                'cholesterol': patient_data.get('cholesterol', 0),
                'fasting_blood_sugar': patient_data.get('fasting_blood_sugar', ''),
                'rest_ecg': patient_data.get('rest_ecg', ''),
                'max_heart_rate': patient_data.get('max_heart_rate', 0),
                'exercise_angina': patient_data.get('exercise_angina', ''),
                'st_depression': patient_data.get('st_depression', 0.0),
                'slope': patient_data.get('slope', ''),
                'colored_vessels': patient_data.get('colored_vessels', 0),
                'thalassemia': patient_data.get('thalassemia', ''),
                'positive_predictions': prediction_results.get('positive_predictions', 0),
                'confidence_level': prediction_results.get('confidence_level', 'Unknown'),
                'overall_result': prediction_results.get('overall_result', ''),
                'recommendation': prediction_results.get('recommendation', ''),
                'model_results': json.dumps(prediction_results.get('results', {})),
                'model_probabilities': json.dumps(prediction_results.get('probabilities', {}))
            }
            
            # Insert into Supabase
            response = requests.post(
                f"{self.supabase_url}/rest/v1/predictions",
                headers=self.headers,
                json=prediction_record,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"✅ Prediction saved to database: {session_id}")
                return session_id
            else:
                print(f"⚠️ Database save failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"⚠️ Database error: {e}")
            return None
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions from database"""
        if not self.connected:
            return []
        
        try:
            response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=*&order=timestamp.desc&limit={limit}",
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return []
                
        except Exception as e:
            print(f"⚠️ Database query error: {e}")
            return []
    
    def get_analytics_data(self):
        """Get analytics data from database"""
        if not self.connected:
            return {}
        
        try:
            # Get basic statistics
            stats_response = requests.get(
                f"{self.supabase_url}/rest/v1/predictions?select=positive_predictions,patient_age,patient_sex",
                headers=self.headers,
                timeout=10
            )
            
            if stats_response.status_code == 200:
                data = stats_response.json()
                
                # Calculate analytics
                total_predictions = len(data)
                high_risk_count = len([p for p in data if p.get('positive_predictions', 0) >= 3])
                avg_age = sum([p.get('patient_age', 0) for p in data]) / max(total_predictions, 1)
                
                return {
                    'total_predictions': total_predictions,
                    'high_risk_predictions': high_risk_count,
                    'risk_percentage': (high_risk_count / max(total_predictions, 1)) * 100,
                    'average_patient_age': round(avg_age, 1)
                }
            else:
                return {}
                
        except Exception as e:
            print(f"⚠️ Analytics error: {e}")
            return {}

# Global database instance
supabase_db = SupabaseDB()

# Convenience functions for easy integration
def save_prediction_to_db(patient_data, prediction_results, model_outputs=None):
    """Save prediction to Supabase database"""
    return supabase_db.save_prediction(patient_data, prediction_results, model_outputs)

def get_recent_predictions_from_db(limit=10):
    """Get recent predictions from database"""
    return supabase_db.get_recent_predictions(limit)

def get_database_analytics():
    """Get analytics from database"""
    return supabase_db.get_analytics_data()

def is_database_connected():
    """Check if database is connected"""
    return supabase_db.connected

def init_database():
    """Initialize database connection"""
    global supabase_db
    supabase_db = SupabaseDB()
    
    if supabase_db.connected:
        print("🗄️ Supabase database connected successfully")
        return True
    else:
        print("⚠️ Supabase database connection failed")
        print("💡 Add SUPABASE_URL and SUPABASE_ANON_KEY to environment variables")
        return False