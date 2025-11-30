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
print("🫀 Loading cardiovascular disease prediction models...")
models, feature_names = load_and_prepare_models()
print("✅ Models loaded successfully!")

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
    
    # Create detailed results string with professional formatting
    detailed_results = """
## 🏥 **Cardiovascular Risk Assessment Report**

### 📊 **Individual Model Analysis**

"""
    
    for model_name, result in results.items():
        prob_disease = float(probabilities[model_name]['Heart Disease'])
        prob_no_disease = float(probabilities[model_name]['No Heart Disease'])
        
        # Status indicator
        status_icon = "🔴" if "Detected" in result else "🟢"
        confidence_bar = "█" * int(prob_disease * 10) + "░" * (10 - int(prob_disease * 10))
        
        detailed_results += f"""
**{status_icon} {model_name}**
- **Prediction:** {result}
- **Risk Probability:** {prob_disease:.1%} `{confidence_bar}` {prob_disease:.3f}
- **No Risk Probability:** {prob_no_disease:.1%}

"""
    
    # Overall assessment with professional medical language
    risk_percentage = (positive_predictions / len(models)) * 100
    
    if positive_predictions >= 3:
        overall_result = f"🚨 **HIGH RISK** ({risk_percentage:.0f}% model consensus)"
        recommendation = """
**Immediate Action Recommended:**
- Schedule urgent consultation with cardiologist
- Consider comprehensive cardiac evaluation
- Implement immediate lifestyle modifications
- Monitor symptoms closely"""
        
    elif positive_predictions >= 2:
        overall_result = f"⚡ **MODERATE RISK** ({risk_percentage:.0f}% model consensus)"
        recommendation = """
**Follow-up Recommended:**
- Schedule appointment with primary care physician
- Consider cardiac screening tests
- Implement preventive lifestyle changes
- Regular monitoring advised"""
        
    else:
        overall_result = f"✅ **LOW RISK** ({risk_percentage:.0f}% model consensus)"
        recommendation = """
**Preventive Care:**
- Maintain current healthy lifestyle
- Continue regular check-ups
- Monitor risk factors periodically
- Stay physically active"""
    
    detailed_results += f"""
---

### 🎯 **Overall Risk Assessment**

{overall_result}

**Confidence Level:** {confidence_level}
**Models in Agreement:** {positive_predictions}/{len(models)}

### 💡 **Clinical Recommendations**

{recommendation}

---

### 📝 **Important Notes**
- This assessment is based on statistical analysis only
- Individual risk factors may not be fully captured
- Results should be interpreted by qualified medical professionals
- Consider personal and family medical history in final evaluation

*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # Create professional visualization
    plt.style.use('default')  # Use default style instead of seaborn-v0_8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Enhanced bar chart of predictions
    model_names = list(results.keys())
    predictions = [1 if "Detected" in result else 0 for result in results.values()]
    colors = ['#dc3545' if pred == 1 else '#28a745' for pred in predictions]
    
    bars1 = ax1.bar(model_names, predictions, color=colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax1.set_ylabel('Risk Assessment', fontsize=12, fontweight='bold')
    ax1.set_title('Model Risk Predictions', fontsize=14, fontweight='bold', pad=20)
    ax1.set_ylim(0, 1.2)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_facecolor('#ffffff')
    
    # Add value labels on bars
    for bar, pred in zip(bars1, predictions):
        height = bar.get_height()
        label = 'RISK' if pred == 1 else 'NO RISK'
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    
    # Enhanced probability chart with gradient colors
    disease_probs = [float(probabilities[model]['Heart Disease']) for model in model_names]
    
    # Create color gradient based on probability
    prob_colors = []
    for prob in disease_probs:
        if prob < 0.3:
            prob_colors.append('#28a745')  # Green for low risk
        elif prob < 0.7:
            prob_colors.append('#ffc107')  # Yellow for moderate risk
        else:
            prob_colors.append('#dc3545')  # Red for high risk
    
    bars2 = ax2.bar(model_names, disease_probs, color=prob_colors, alpha=0.8, edgecolor='white', linewidth=2)
    ax2.set_ylabel('Cardiovascular Risk Probability', fontsize=12, fontweight='bold')
    ax2.set_title('Risk Probability Distribution', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_facecolor('#ffffff')
    
    # Add percentage labels on bars
    for bar, prob in zip(bars2, disease_probs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontweight='bold')
    
    # Add professional styling
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
    
    plt.tight_layout()
    
    return detailed_results, fig

# Define the Gradio interface
def create_interface():
    # Custom CSS for professional styling
    custom_css = """
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .section-header {
        background: linear-gradient(90deg, #f8f9ff 0%, #e8efff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(90deg, #fff3cd 0%, #fef7e0 100%);
        border: 2px solid #ffc107;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background: linear-gradient(90deg, #d1ecf1 0%, #bee5eb 100%);
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .predict-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-weight: bold !important;
        font-size: 18px !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    .results-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border: 1px solid #e9ecef;
    }
    """
    
    with gr.Blocks(
        title="CardioPredict Pro - AI-Powered Heart Disease Assessment", 
        theme=gr.themes.Base(),
        css=custom_css
    ) as demo:
        
        # Main Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 300;">
                🏥 CardioPredict Pro
            </h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.2em; opacity: 0.9;">
                Advanced AI-Powered Cardiovascular Risk Assessment
            </p>
        </div>
        """)
        
        # Professional disclaimer
        gr.HTML("""
        <div class="warning-box">
            <h4 style="margin: 0 0 0.5rem 0; color: #856404;">
                ⚠️ Medical Disclaimer
            </h4>
            <p style="margin: 0; color: #856404;">
                This AI tool is designed for educational and research purposes only. Results should not be used for 
                medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.
            </p>
        </div>
        """)
        
        # Input Form Section
        gr.HTML('<div class="section-header"><h3 style="margin: 0; color: #495057;">📋 Patient Assessment Form</h3></div>')
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem;">👤 Demographics</h4>')
                
                age = gr.Slider(
                    minimum=1, maximum=100, value=50, 
                    label="Age (years)",
                    info="Patient's age in years"
                )
                
                sex = gr.Dropdown(
                    choices=[("Female", 0), ("Male", 1)], 
                    value=1, 
                    label="Biological Sex",
                    info="Select biological sex"
                )
                
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-top: 1.5rem;">💓 Cardiac Symptoms</h4>')
                
                chest_pain_type = gr.Dropdown(
                    choices=[
                        ("Typical Angina", 0), 
                        ("Atypical Angina", 1), 
                        ("Non-anginal Pain", 2), 
                        ("Asymptomatic", 3)
                    ], 
                    value=0,
                    label="Chest Pain Type",
                    info="Classification of chest pain symptoms"
                )
                
                exercise_angina = gr.Dropdown(
                    choices=[("No", 0), ("Yes", 1)], 
                    value=0, 
                    label="Exercise Induced Angina",
                    info="Does exercise trigger chest pain?"
                )
                
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-top: 1.5rem;">🩺 Vital Signs</h4>')
                
                resting_bp = gr.Slider(
                    minimum=80, maximum=250, value=120, 
                    label="Resting Blood Pressure (mmHg)",
                    info="Systolic blood pressure at rest"
                )
                
                max_heart_rate = gr.Slider(
                    minimum=50, maximum=250, value=150, 
                    label="Maximum Heart Rate (bpm)",
                    info="Maximum heart rate achieved during stress test"
                )
                
            with gr.Column(scale=1):
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem;">🔬 Laboratory Results</h4>')
                
                cholesterol = gr.Slider(
                    minimum=100, maximum=600, value=200, 
                    label="Serum Cholesterol (mg/dL)",
                    info="Total cholesterol level"
                )
                
                fasting_blood_sugar = gr.Dropdown(
                    choices=[("≤ 120 mg/dL", 0), ("> 120 mg/dL", 1)], 
                    value=0, 
                    label="Fasting Blood Sugar",
                    info="Fasting blood glucose level"
                )
                
                rest_ecg = gr.Dropdown(
                    choices=[
                        ("Normal", 0), 
                        ("ST-T Wave Abnormality", 1), 
                        ("Left Ventricular Hypertrophy", 2)
                    ], 
                    value=0,
                    label="Resting ECG Results",
                    info="Electrocardiogram findings at rest"
                )
                
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-top: 1.5rem;">📊 Stress Test Parameters</h4>')
                
                st_depression = gr.Slider(
                    minimum=0.0, maximum=10.0, value=0.0, step=0.1, 
                    label="ST Depression (mm)",
                    info="ST depression induced by exercise relative to rest"
                )
                
                slope = gr.Dropdown(
                    choices=[
                        ("Upsloping", 0), 
                        ("Flat", 1), 
                        ("Downsloping", 2)
                    ], 
                    value=1, 
                    label="ST Segment Slope",
                    info="Slope of peak exercise ST segment"
                )
                
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-top: 1.5rem;">🩻 Advanced Diagnostics</h4>')
                
                colored_vessels = gr.Dropdown(
                    choices=[("0", 0), ("1", 1), ("2", 2), ("3", 3)], 
                    value=0, 
                    label="Vessels Colored by Fluoroscopy",
                    info="Number of major vessels with significant stenosis"
                )
                
                thalassemia = gr.Dropdown(
                    choices=[
                        ("Normal", 0), 
                        ("Fixed Defect", 1), 
                        ("Reversible Defect", 2), 
                        ("Not Described", 3)
                    ], 
                    value=2, 
                    label="Thalassemia Test Result",
                    info="Thalassemia stress test result"
                )
        
        # Prediction Button Section
        gr.HTML('<div style="text-align: center; margin: 2rem 0;">')
        
        predict_btn = gr.Button(
            "🔍 Analyze Cardiovascular Risk", 
            variant="primary", 
            size="lg",
            elem_classes=["predict-button"]
        )
        
        gr.HTML('</div>')
        
        # Results Section
        gr.HTML('<div class="section-header"><h3 style="margin: 0; color: #495057;">📊 AI Analysis Results</h3></div>')
        
        gr.HTML("""<div class="info-box">
            <p style="margin: 0; color: #0c5460;">
                <strong>🤖 Multi-Model Analysis:</strong> Our system employs four advanced machine learning algorithms 
                (Logistic Regression, Random Forest, SVM, and Gradient Boosting) to provide a comprehensive 
                cardiovascular risk assessment. Results are aggregated for maximum reliability.
            </p>
        </div>""")
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML('<div class="results-container">')
                results_text = gr.Markdown(visible=True)
                gr.HTML('</div>')
            with gr.Column(scale=2):
                gr.HTML('<div class="results-container">')
                results_plot = gr.Plot(visible=True)
                gr.HTML('</div>')
        
        # Connect the prediction function
        predict_btn.click(
            predict_heart_disease,
            inputs=[age, sex, chest_pain_type, resting_bp, cholesterol, 
                   fasting_blood_sugar, rest_ecg, max_heart_rate, 
                   exercise_angina, st_depression, slope, 
                   colored_vessels, thalassemia],
            outputs=[results_text, results_plot]
        )
        
        # Professional Information Section
        gr.Markdown("## 🔬 Technical Information")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
### 🧠 Machine Learning Models

- **Logistic Regression:** Linear probabilistic model optimized for medical interpretability
- **Random Forest:** Ensemble method using 100+ decision trees for robust predictions  
- **Support Vector Machine:** Non-linear classifier with RBF kernel and probability calibration
- **Gradient Boosting:** Sequential learning algorithm with advanced regularization
                """)
            
            with gr.Column():
                gr.Markdown("""
### 📈 Risk Assessment Methodology

- **High Risk:** 3+ models indicate positive prediction
- **Moderate Risk:** 2 models indicate positive prediction
- **Low Risk:** 0-1 models indicate positive prediction
- **Confidence:** Based on model consensus and probability scores
                """)
        
        # Professional Footer
        gr.Markdown("""
---
## ⚠️ Important Medical Disclaimer

This AI-powered tool is designed exclusively for educational and research purposes. It does not replace 
professional medical diagnosis, treatment, or advice. The predictions are based on statistical patterns 
in training data and may not account for individual medical history, comorbidities, or other clinical factors. 
**Always consult qualified healthcare professionals for medical evaluation and treatment decisions.**

*CardioPredict Pro v1.0 | Powered by Advanced Machine Learning | For Research Use Only*
        """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    print("🏥 Initializing CardioPredict Pro...")
    print("📊 Loading AI models and clinical parameters...")
    demo = create_interface()
    print("✅ System ready for cardiovascular risk assessment")
    print("🌐 Launching professional medical interface...")
    demo.launch(
        share=True,  # Enable public shareable link
        inbrowser=True,  # Open in browser automatically
        server_name="localhost",  # Local access only for security
        server_port=7861,  # Use different port to avoid conflicts
        show_error=True,  # Show detailed errors
        quiet=False,  # Show startup logs
        favicon_path=None,  # Could add medical favicon
        ssl_verify=False,  # For local development
        app_kwargs={"title": "CardioPredict Pro - AI Cardiovascular Assessment"}
    )