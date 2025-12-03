-- CardioPredict Pro Database Schema for Supabase
-- Run this SQL in your Supabase SQL Editor to create the required tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create predictions table
CREATE TABLE public.predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Patient Information
    patient_name TEXT NOT NULL,
    patient_age INTEGER NOT NULL CHECK (patient_age > 0 AND patient_age <= 150),
    patient_sex TEXT NOT NULL CHECK (patient_sex IN ('Male', 'Female', 'Unknown')),
    
    -- Clinical Parameters
    chest_pain_type TEXT NOT NULL,
    resting_bp INTEGER NOT NULL CHECK (resting_bp > 0),
    cholesterol INTEGER NOT NULL CHECK (cholesterol > 0),
    fasting_blood_sugar TEXT NOT NULL,
    rest_ecg TEXT NOT NULL,
    max_heart_rate INTEGER NOT NULL CHECK (max_heart_rate > 0),
    exercise_angina TEXT NOT NULL,
    st_depression DECIMAL(4,2) NOT NULL CHECK (st_depression >= 0),
    slope TEXT NOT NULL,
    colored_vessels INTEGER NOT NULL CHECK (colored_vessels >= 0 AND colored_vessels <= 4),
    thalassemia TEXT NOT NULL,
    
    -- AI Prediction Results
    positive_predictions INTEGER NOT NULL CHECK (positive_predictions >= 0 AND positive_predictions <= 4),
    confidence_level TEXT NOT NULL CHECK (confidence_level IN ('High', 'Medium', 'Low')),
    overall_result TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    model_results JSONB NOT NULL,
    model_probabilities JSONB NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX idx_predictions_timestamp ON public.predictions(timestamp DESC);
CREATE INDEX idx_predictions_patient_age ON public.predictions(patient_age);
CREATE INDEX idx_predictions_risk_level ON public.predictions(positive_predictions);
CREATE INDEX idx_predictions_confidence ON public.predictions(confidence_level);

-- Create a view for analytics
CREATE VIEW public.prediction_analytics AS
SELECT 
    DATE_TRUNC('day', timestamp) as prediction_date,
    COUNT(*) as total_predictions,
    COUNT(*) FILTER (WHERE positive_predictions >= 3) as high_risk_predictions,
    COUNT(*) FILTER (WHERE positive_predictions = 2) as moderate_risk_predictions,
    COUNT(*) FILTER (WHERE positive_predictions <= 1) as low_risk_predictions,
    AVG(patient_age) as avg_patient_age,
    COUNT(*) FILTER (WHERE patient_sex = 'Male') as male_patients,
    COUNT(*) FILTER (WHERE patient_sex = 'Female') as female_patients
FROM public.predictions
GROUP BY DATE_TRUNC('day', timestamp)
ORDER BY prediction_date DESC;

-- Enable Row Level Security (RLS) for better security
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- Create policy that allows anonymous access (adjust as needed)
CREATE POLICY "Enable read access for all users" ON public.predictions
    FOR SELECT USING (true);

CREATE POLICY "Enable insert access for all users" ON public.predictions
    FOR INSERT WITH CHECK (true);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for updated_at
CREATE TRIGGER handle_updated_at
    BEFORE UPDATE ON public.predictions
    FOR EACH ROW
    EXECUTE FUNCTION public.handle_updated_at();

-- Create a function for quick analytics
CREATE OR REPLACE FUNCTION public.get_prediction_stats()
RETURNS TABLE (
    total_predictions BIGINT,
    high_risk_count BIGINT,
    moderate_risk_count BIGINT,
    low_risk_count BIGINT,
    avg_age NUMERIC,
    male_count BIGINT,
    female_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_predictions,
        COUNT(*) FILTER (WHERE positive_predictions >= 3) as high_risk_count,
        COUNT(*) FILTER (WHERE positive_predictions = 2) as moderate_risk_count,
        COUNT(*) FILTER (WHERE positive_predictions <= 1) as low_risk_count,
        ROUND(AVG(patient_age), 1) as avg_age,
        COUNT(*) FILTER (WHERE patient_sex = 'Male') as male_count,
        COUNT(*) FILTER (WHERE patient_sex = 'Female') as female_count
    FROM public.predictions;
END;
$$ language 'plpgsql';

-- Insert sample data for testing (optional)
-- INSERT INTO public.predictions (
--     patient_name, patient_age, patient_sex, chest_pain_type, resting_bp, cholesterol,
--     fasting_blood_sugar, rest_ecg, max_heart_rate, exercise_angina, st_depression,
--     slope, colored_vessels, thalassemia, positive_predictions, confidence_level,
--     overall_result, recommendation, model_results, model_probabilities
-- ) VALUES (
--     'Test Patient', 45, 'Male', 'Typical Angina', 120, 200,
--     '≤ 120 mg/dL', 'Normal', 150, 'No', 0.0,
--     'Upsloping', 0, 'Normal', 1, 'Medium',
--     'LOW RISK: Models suggest lower probability of heart disease',
--     'Maintain a healthy lifestyle and regular check-ups',
--     '{"Logistic Regression": "No Heart Disease", "Random Forest": "Heart Disease Detected"}',
--     '{"Logistic Regression": {"No Heart Disease": "0.65", "Heart Disease": "0.35"}}'
-- );

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO anon;
GRANT SELECT, INSERT ON public.predictions TO anon;
GRANT SELECT ON public.prediction_analytics TO anon;
GRANT EXECUTE ON FUNCTION public.get_prediction_stats() TO anon;

-- Comment for documentation
COMMENT ON TABLE public.predictions IS 'Stores all cardiovascular risk assessments from CardioPredict Pro';
COMMENT ON VIEW public.prediction_analytics IS 'Aggregated analytics view for prediction trends and statistics';
COMMENT ON FUNCTION public.get_prediction_stats() IS 'Quick function to get overall prediction statistics';