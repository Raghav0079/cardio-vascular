import gradio as gr
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load and prepare the dataset
def load_and_prepare_models():
    """Load the dataset and train all models"""
    # Load the dataset (you'll need to adjust the path)
    try:
        df = pd.read_csv('Cardio_vascular.csv')
    except:
        # If CSV not found, create a sample dataset for demo
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'age': np.random.randint(29, 80, n_samples),
            'sex': np.random.randint(0, 2, n_samples),
            'cp': np.random.randint(0, 4, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(126, 564, n_samples),
            'fbs': np.random.randint(0, 2, n_samples),
            'restecg': np.random.randint(0, 3, n_samples),
            'thalach': np.random.randint(71, 202, n_samples),
            'exang': np.random.randint(0, 2, n_samples),
            'oldpeak': np.random.uniform(0, 6.2, n_samples),
            'slope': np.random.randint(0, 3, n_samples),
            'ca': np.random.randint(0, 4, n_samples),
            'thal': np.random.randint(0, 4, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train all models
    models = {}
    
    # Logistic Regression (tuned)
    models['Logistic Regression'] = LogisticRegression(C=1, solver='liblinear', random_state=42)
    models['Logistic Regression'].fit(X_train, y_train)
    
    # Random Forest
    models['Random Forest'] = RandomForestClassifier(random_state=42)
    models['Random Forest'].fit(X_train, y_train)
    
    # SVM
    models['SVM'] = SVC(probability=True, random_state=42)
    models['SVM'].fit(X_train, y_train)
    
    # Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingClassifier(random_state=42)
    models['Gradient Boosting'].fit(X_train, y_train)
    
    return models, X.columns.tolist()

# Load models and feature names
models, feature_names = load_and_prepare_models()

def predict_heart_disease(age, sex, chest_pain_type, resting_bp, cholesterol, 
                         fasting_blood_sugar, rest_ecg, max_heart_rate, 
                         exercise_angina, st_depression, slope, 
                         colored_vessels, thalassemia):
    """Make predictions using all models"""
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [chest_pain_type],
        'trestbps': [resting_bp],
        'chol': [cholesterol],
        'fbs': [fasting_blood_sugar],
        'restecg': [rest_ecg],
        'thalach': [max_heart_rate],
        'exang': [exercise_angina],
        'oldpeak': [st_depression],
        'slope': [slope],
        'ca': [colored_vessels],
        'thal': [thalassemia]
    })
    
    # Get predictions from all models
    results = {}
    probabilities = {}
    
    for model_name, model in models.items():
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        results[model_name] = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
        probabilities[model_name] = {
            'No Heart Disease': f"{probability[0]:.3f}",
            'Heart Disease': f"{probability[1]:.3f}"
        }
    
    # Create summary
    positive_predictions = sum(1 for model_name, result in results.items() if "Detected" in result)
    confidence_level = "High" if positive_predictions >= 3 or positive_predictions == 0 else "Medium"
    
    if positive_predictions >= 3:
        overall_result = "⚠️ HIGH RISK: Multiple models indicate potential heart disease"
        recommendation = "Please consult with a healthcare professional immediately for proper evaluation."
    elif positive_predictions >= 2:
        overall_result = "⚡ MODERATE RISK: Some models indicate potential concerns"
        recommendation = "Consider scheduling a check-up with your healthcare provider."
    else:
        overall_result = "✅ LOW RISK: Models suggest lower probability of heart disease"
        recommendation = "Maintain a healthy lifestyle and regular check-ups."
    
    # Create detailed results string
    detailed_results = "**Individual Model Predictions:**\n\n"
    for model_name, result in results.items():
        prob_disease = float(probabilities[model_name]['Heart Disease'])
        detailed_results += f"• **{model_name}**: {result} (Confidence: {prob_disease:.1%})\n"
    
    detailed_results += f"\n**Overall Assessment:** {overall_result}\n"
    detailed_results += f"**Confidence Level:** {confidence_level}\n"
    detailed_results += f"**Recommendation:** {recommendation}\n"
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of predictions
    model_names = list(results.keys())
    predictions = [1 if "Detected" in result else 0 for result in results.values()]
    colors = ['red' if pred == 1 else 'green' for pred in predictions]
    
    ax1.bar(model_names, predictions, color=colors, alpha=0.7)
    ax1.set_ylabel('Prediction (1=Disease, 0=No Disease)')
    ax1.set_title('Model Predictions')
    ax1.set_ylim(0, 1.2)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Probability chart
    disease_probs = [float(probabilities[model]['Heart Disease']) for model in model_names]
    ax2.bar(model_names, disease_probs, color='orange', alpha=0.7)
    ax2.set_ylabel('Probability of Heart Disease')
    ax2.set_title('Disease Probability by Model')
    ax2.set_ylim(0, 1)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    return detailed_results, fig

# Define the Gradio interface
def create_interface():
    with gr.Blocks(title="🫀 Cardiovascular Disease Prediction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🫀 Cardiovascular Disease Prediction System
        
        This tool uses multiple machine learning models to assess cardiovascular disease risk based on your health parameters.
        **Note: This is for educational purposes only and should not replace professional medical advice.**
        """)
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 👤 Personal Information")
                age = gr.Slider(minimum=1, maximum=100, value=50, label="Age")
                sex = gr.Radio(choices=[0, 1], value=1, label="Sex (0: Female, 1: Male)")
                
                gr.Markdown("### 💓 Heart-Related Symptoms")
                chest_pain_type = gr.Radio(
                    choices=[0, 1, 2, 3], 
                    value=0,
                    label="Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)"
                )
                exercise_angina = gr.Radio(choices=[0, 1], value=0, label="Exercise Induced Angina (0: No, 1: Yes)")
                
                gr.Markdown("### 🩺 Clinical Measurements")
                resting_bp = gr.Slider(minimum=80, maximum=250, value=120, label="Resting Blood Pressure (mm Hg)")
                cholesterol = gr.Slider(minimum=100, maximum=600, value=200, label="Cholesterol Level (mg/dl)")
                max_heart_rate = gr.Slider(minimum=50, maximum=250, value=150, label="Maximum Heart Rate Achieved")
                
            with gr.Column():
                gr.Markdown("### 🔬 Lab Results")
                fasting_blood_sugar = gr.Radio(choices=[0, 1], value=0, label="Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)")
                
                rest_ecg = gr.Radio(
                    choices=[0, 1, 2], 
                    value=0,
                    label="Resting ECG Results (0: Normal, 1: ST-T Wave Abnormality, 2: Left Ventricular Hypertrophy)"
                )
                
                gr.Markdown("### 📊 Additional Parameters")
                st_depression = gr.Slider(minimum=0.0, maximum=10.0, value=0.0, step=0.1, label="ST Depression Induced by Exercise")
                slope = gr.Radio(choices=[0, 1, 2], value=1, label="Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)")
                colored_vessels = gr.Radio(choices=[0, 1, 2, 3], value=0, label="Number of Major Vessels Colored by Fluoroscopy (0-3)")
                thalassemia = gr.Radio(choices=[0, 1, 2, 3], value=2, label="Thalassemia (0: Normal, 1: Fixed Defect, 2: Reversible Defect, 3: Not described)")
        
        predict_btn = gr.Button("🔍 Predict Heart Disease Risk", variant="primary", size="lg")
        
        gr.Markdown("### 📊 Prediction Results")
        
        with gr.Row():
            with gr.Column():
                results_text = gr.Markdown()
            with gr.Column():
                results_plot = gr.Plot()
        
        # Connect the prediction function
        predict_btn.click(
            predict_heart_disease,
            inputs=[age, sex, chest_pain_type, resting_bp, cholesterol, 
                   fasting_blood_sugar, rest_ecg, max_heart_rate, 
                   exercise_angina, st_depression, slope, 
                   colored_vessels, thalassemia],
            outputs=[results_text, results_plot]
        )
        
        gr.Markdown("""
        ### ℹ️ About the Models
        - **Logistic Regression**: Linear model optimized for interpretability
        - **Random Forest**: Ensemble method using multiple decision trees
        - **SVM**: Support Vector Machine with probability estimation
        - **Gradient Boosting**: Advanced ensemble method with sequential learning
        
        ### ⚠️ Disclaimer
        This tool is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.
        """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)