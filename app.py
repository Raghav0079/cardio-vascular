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
from datetime import datetime
import os
import tempfile
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.graphics.shapes import Drawing, Rect
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from io import BytesIO
warnings.filterwarnings('ignore')

# Load environment variables
def load_env_vars():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Load environment variables
load_env_vars()

# Optional WandB integration - gracefully handles missing package or API key
try:
    from wandb_integration import init_wandb_tracking, log_training_data, log_prediction_data, create_wandb_dashboard, finish_wandb_run
    WANDB_AVAILABLE = True
    print("📊 WandB integration available")
except ImportError:
    WANDB_AVAILABLE = False
    print("ℹ️ WandB not available - continuing without tracking")
    # Define dummy functions
    def init_wandb_tracking(*args, **kwargs): return False
    def log_training_data(*args, **kwargs): pass
    def log_prediction_data(*args, **kwargs): pass 
    def create_wandb_dashboard(*args, **kwargs): pass
    def finish_wandb_run(*args, **kwargs): pass

def load_and_prepare_models():
    """Load the dataset and train all models"""
    print("Loading cardiovascular disease prediction models...")
    
    # Initialize optional WandB tracking
    wandb_initialized = False
    if WANDB_AVAILABLE:
        try:
            wandb_initialized = init_wandb_tracking()
            if wandb_initialized:
                print("📊 WandB tracking initialized successfully")
        except Exception as e:
            print(f"⚠️ WandB tracking failed to initialize: {e}")
    
    # Create synthetic dataset for demo
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
    
    # Log model training to wandb
    try:
        if wandb.run is not None:
            wandb.log({
                "training/dataset_size": len(df),
                "training/features": len(X.columns),
                "training/models_trained": len(models),
                "training/test_split": 0.2
            })
            # Log basic model info
            for model_name in models.keys():
                wandb.log({f"models/{model_name.lower().replace(' ', '_')}_trained": True})
    except Exception as e:
        print(f"WandB model logging failed: {e}")
    
    print("Models loaded successfully!")
    
    # Log training data to WandB if available
    if WANDB_AVAILABLE and wandb_initialized:
        try:
            log_training_data(models, df)
            print("📈 Training data logged to WandB")
        except Exception as e:
            print(f"⚠️ WandB training logging failed: {e}")
    
    return models, X.columns.tolist()

# Load models and feature names
print("Loading cardiovascular disease prediction models...")
models, feature_names = load_and_prepare_models()
print("Models loaded successfully!")

def generate_pdf_report(report_data, input_data):
    """Generate a professional medical-grade PDF report for cardiovascular assessment"""
    
    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf_path = tmp_file.name
    
    # Create the PDF document with professional margins
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=50, leftMargin=50, 
                           topMargin=50, bottomMargin=50)
    
    # Define professional medical styles - optimized for smaller PDF
    styles = getSampleStyleSheet()
    
    # Medical Header Style - reduced size
    medical_title = ParagraphStyle(
        'MedicalTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        spaceBefore=8,
        textColor=colors.HexColor('#1565C0'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#1565C0'),
        borderPadding=6
    )
    
    # Institution/Facility Style - reduced
    facility_style = ParagraphStyle(
        'FacilityStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        textColor=colors.HexColor('#424242'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    # Section Header Style - compact
    section_header = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=8,
        textColor=colors.HexColor('#0D47A1'),
        fontName='Helvetica-Bold',
        borderWidth=0.5,
        borderColor=colors.HexColor('#0D47A1'),
        leftIndent=0,
        borderPadding=3
    )
    
    # Subsection Style - compact
    subsection_style = ParagraphStyle(
        'SubsectionStyle',
        parent=styles['Heading3'],
        fontSize=10,
        spaceAfter=4,
        spaceBefore=6,
        textColor=colors.HexColor('#1976D2'),
        fontName='Helvetica-Bold'
    )
    
    # Clinical Text Style - compact
    clinical_style = ParagraphStyle(
        'ClinicalStyle',
        parent=styles['Normal'],
        fontSize=9,
        spaceAfter=4,
        fontName='Helvetica',
        leftIndent=8
    )
    
    # Important Notice Style - compact
    notice_style = ParagraphStyle(
        'NoticeStyle',
        parent=styles['Normal'],
        fontSize=8,
        spaceAfter=6,
        textColor=colors.HexColor('#D32F2F'),
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=colors.HexColor('#D32F2F'),
        borderPadding=4,
        backColor=colors.HexColor('#FFEBEE')
    )
    
    # Build the professional medical report - optimized
    story = []
    
    # Medical Report Header with Professional Branding - compact
    story.append(Paragraph("CARDIOVASCULAR RISK ASSESSMENT", medical_title))
    story.append(Paragraph("CardioPredict Pro Medical AI System", facility_style))
    story.append(Spacer(1, 8))
    
    # Compact Medical Watermark
    story.append(Paragraph(
        "<i>CONFIDENTIAL MEDICAL REPORT</i>", 
        ParagraphStyle('Watermark', parent=styles['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER, fontName='Helvetica-Oblique')
    ))
    story.append(Spacer(1, 10))
    
    # Report Information Header - compact
    report_info = [
        ['REPORT INFORMATION', ''],
        ['Report ID:', f"CVD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"],
        ['Generated:', datetime.now().strftime('%B %d, %Y at %H:%M')],
        ['AI Version:', 'CardioPredict Pro v1.0']
    ]
    
    report_table = Table(report_info, colWidths=[2.2*inch, 4.3*inch])
    report_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#0D47A1')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 0), (-1, 0), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F5F5F5')])
    ]))
    
    story.append(report_table)
    story.append(Spacer(1, 10))
    
    # Patient Demographics Section - compact
    story.append(Paragraph("I. PATIENT DEMOGRAPHICS", section_header))
    story.append(Spacer(1, 4))
    
    patient_info = [
        ['PATIENT IDENTIFICATION', ''],
        ['Name:', report_data['patient_name']],
        ['Age:', f"{report_data['age']} years"],
        ['Sex:', report_data['sex']],
        ['Assessment Date:', report_data['timestamp']]
    ]
    
    patient_table = Table(patient_info, colWidths=[2.2*inch, 4.3*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#1976D2')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#E3F2FD')])
    ]))
    
    story.append(patient_table)
    story.append(Spacer(1, 8))
    
    # Clinical Assessment Parameters Section - consolidated
    story.append(Paragraph("II. CLINICAL PARAMETERS", section_header))
    story.append(Spacer(1, 4))
    
    # Consolidated clinical data in one table
    clinical_data = [
        ['PARAMETER', 'VALUE', 'REFERENCE'],
        ['Age', f"{input_data['age'].iloc[0]} years", 'Adult'],
        ['Sex', 'Male' if input_data['sex'].iloc[0] == 1 else 'Female', 'M/F'],
        ['Chest Pain', ['Typical Angina', 'Atypical Angina', 'Non-anginal', 'Asymptomatic'][input_data['cp'].iloc[0]], 'Symptom type'],
        ['Resting BP', f"{input_data['trestbps'].iloc[0]} mmHg", '<120/80'],
        ['Cholesterol', f"{input_data['chol'].iloc[0]} mg/dL", '<200'],
        ['Fasting Glucose', '>120' if input_data['fbs'].iloc[0] == 1 else '≤120', '<100 mg/dL'],
        ['Resting ECG', ['Normal', 'ST-T Abnormal', 'LVH'][input_data['restecg'].iloc[0]], 'Rhythm'],
        ['Max Heart Rate', f"{input_data['thalach'].iloc[0]} bpm", f">{220-input_data['age'].iloc[0]}"],
        ['Exercise Angina', 'Yes' if input_data['exang'].iloc[0] == 1 else 'No', 'Present/Absent'],
        ['ST Depression', f"{input_data['oldpeak'].iloc[0]} mm", 'Exercise change'],
        ['ST Slope', ['Up', 'Flat', 'Down'][input_data['slope'].iloc[0]], 'Exercise pattern'],
        ['Vessels (Fluoro)', f"{input_data['ca'].iloc[0]}", 'Major vessels'],
        ['Thalassemia', ['Normal', 'Fixed', 'Reversible', 'N/A'][input_data['thal'].iloc[0]], 'Perfusion']
    ]
    
    clinical_table = Table(clinical_data, colWidths=[2.2*inch, 2.2*inch, 2.1*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F8E9')])
    ]))
    
    story.append(clinical_table)
    story.append(Spacer(1, 10))
    
    # AI Analysis Results Section - simplified
    story.append(Paragraph("III. AI ANALYSIS", section_header))
    story.append(Spacer(1, 4))
    
    model_data = [['MODEL', 'PREDICTION', 'PROBABILITY']]
    
    for model_name, result in report_data['results'].items():
        risk_prob = float(report_data['probabilities'][model_name]['Heart Disease'])
        prediction = "POSITIVE" if "Detected" in result else "NEGATIVE"
        model_data.append([model_name, prediction, f"{risk_prob:.1%}"])
    
    model_table = Table(model_data, colWidths=[2.2*inch, 2.2*inch, 2.1*inch])
    model_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (2, 0), colors.HexColor('#1565C0')),
        ('TEXTCOLOR', (0, 0), (2, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    
    # Color code the predictions
    for i, (model_name, result) in enumerate(report_data['results'].items()):
        row_idx = i + 1
        if "Detected" in result:
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (1, row_idx), (1, row_idx), colors.HexColor('#FFCDD2')),
                ('TEXTCOLOR', (1, row_idx), (1, row_idx), colors.HexColor('#C62828'))
            ]))
        else:
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (1, row_idx), (1, row_idx), colors.HexColor('#C8E6C9')),
                ('TEXTCOLOR', (1, row_idx), (1, row_idx), colors.HexColor('#2E7D32'))
            ]))
    
    story.append(model_table)
    story.append(Spacer(1, 8))
    
    # Clinical Consensus - compact
    consensus = f"{report_data['positive_predictions']}/{len(report_data['results'])}"
    risk_level = "HIGH" if report_data['positive_predictions'] >= 3 else "MODERATE" if report_data['positive_predictions'] >= 2 else "LOW"
    
    consensus_data = [
        ['CONSENSUS RESULT', 'VALUE'],
        ['Risk Level', f"{risk_level} RISK"],
        ['Model Agreement', f"{consensus} models positive"],
        ['Confidence', report_data['confidence_level']]
    ]
    
    consensus_table = Table(consensus_data, colWidths=[3.25*inch, 3.25*inch])
    consensus_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#424242')),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    story.append(consensus_table)
    story.append(Spacer(1, 10))
    
    # Clinical Recommendations - compact
    story.append(Paragraph("IV. RECOMMENDATIONS", section_header))
    story.append(Spacer(1, 4))
    
    # Simplified recommendations based on risk level
    if report_data['positive_predictions'] >= 3:
        rec_text = "HIGH RISK: Immediate cardiology consultation recommended. Consider comprehensive cardiac evaluation and risk factor optimization."
    elif report_data['positive_predictions'] >= 2:
        rec_text = "MODERATE RISK: Schedule cardiology consultation within 2-4 weeks. Implement lifestyle modifications and monitor risk factors."
    else:
        rec_text = "LOW RISK: Continue routine preventive care. Maintain healthy lifestyle and regular cardiovascular screening."
    
    story.append(Paragraph(rec_text, clinical_style))
    story.append(Spacer(1, 10))
    
    # Simplified Medical Disclaimer
    story.append(Paragraph("V. DISCLAIMER", section_header))
    story.append(Spacer(1, 4))
    
    disclaimer_text = """
    This AI assessment is for educational purposes only and does not replace professional 
    medical diagnosis. Results should be interpreted by qualified healthcare professionals. 
    Always consult your physician for medical advice and treatment decisions.
    """
    
    story.append(Paragraph(disclaimer_text, notice_style))
    story.append(Spacer(1, 10))
    
    # Simple Footer
    footer_data = [
        ['Report ID:', f"CVD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"],
        ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['System:', 'CardioPredict Pro v1.0']
    ]
    
    footer_table = Table(footer_data, colWidths=[2.2*inch, 4.3*inch])
    footer_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('GRID', (0, 0), (-1, -1), 0.3, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
    ]))
    
    story.append(footer_table)
    
    # Build the PDF
    doc.build(story)
    
    return pdf_path

def predict_heart_disease(patient_name, age, sex, chest_pain_type, resting_bp, cholesterol, 
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
    
    # Prepare report data for PDF generation (combine all data into report_data)
    report_data = {
        'patient_name': patient_name,
        'age': age,
        'sex': 'Male' if sex == 1 else 'Female',
        'chest_pain_type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][chest_pain_type],
        'resting_bp': resting_bp,
        'cholesterol': cholesterol,
        'fasting_blood_sugar': '> 120 mg/dL' if fasting_blood_sugar == 1 else '≤ 120 mg/dL',
        'rest_ecg': ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'][rest_ecg],
        'max_heart_rate': max_heart_rate,
        'exercise_angina': 'Yes' if exercise_angina == 1 else 'No',
        'st_depression': st_depression,
        'slope': ['Upsloping', 'Flat', 'Downsloping'][slope],
        'colored_vessels': colored_vessels,
        'thalassemia': ['Normal', 'Fixed Defect', 'Reversible Defect', 'Not Described'][thalassemia],
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': results,
        'probabilities': probabilities,
        'overall_result': overall_result,
        'recommendation': recommendation,
        'positive_predictions': positive_predictions,
        'confidence_level': confidence_level
    }
    
    # Prepare input data for clinical parameters section
    input_data_for_pdf = pd.DataFrame({
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
    
    # Generate PDF report
    pdf_path = generate_pdf_report(report_data, input_data_for_pdf)
    
    # Log prediction to WandB if available
    if WANDB_AVAILABLE:
        try:
            # Prepare patient data for WandB logging (anonymized)
            patient_data_for_wandb = {
                'age': age,
                'sex': 'Male' if sex == 1 else 'Female',
                'chest_pain_type': ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'][chest_pain_type],
                'resting_bp': resting_bp,
                'cholesterol': cholesterol,
                'exercise_angina': 'Yes' if exercise_angina == 1 else 'No',
                'fasting_blood_sugar': '> 120 mg/dL' if fasting_blood_sugar == 1 else 'No',
                'max_heart_rate': max_heart_rate
            }
            
            log_prediction_data(patient_data_for_wandb, report_data)
            create_wandb_dashboard()
        except Exception as e:
            print(f"⚠️ WandB prediction logging failed: {e}")
    
    return detailed_results, fig, pdf_path

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
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem;">👤 Patient Information</h4>')
                
                patient_name = gr.Textbox(
                    label="Patient Name",
                    placeholder="Enter patient's full name",
                    info="Patient's name for medical records",
                    value="",
                    elem_id="patient_name_input",
                    interactive=True,
                    type="text"
                )
                
                gr.HTML('<h4 style="color: #6c757d; border-bottom: 2px solid #e9ecef; padding-bottom: 0.5rem; margin-top: 1.5rem;">👤 Demographics</h4>')
                
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
        
        # PDF Download Section
        gr.HTML('<div class="section-header"><h3 style="margin: 0; color: #495057;">📄 Medical Report Download</h3></div>')
        
        with gr.Row():
            with gr.Column():
                pdf_download = gr.File(
                    label="📋 Professional Medical Report (PDF)",
                    interactive=False,
                    visible=True
                )
                
                gr.HTML("""<div class="info-box" style="margin-top: 1rem;">
                    <p style="margin: 0; color: #0c5460;">
                        <strong>📄 Professional Report:</strong> Complete cardiovascular assessment report 
                        including patient demographics, clinical parameters, AI analysis results, and 
                        medical recommendations formatted for healthcare professionals.
                    </p>
                </div>""")
        
        # Connect the prediction function
        predict_btn.click(
            predict_heart_disease,
            inputs=[patient_name, age, sex, chest_pain_type, resting_bp, cholesterol, 
                   fasting_blood_sugar, rest_ecg, max_heart_rate, 
                   exercise_angina, st_depression, slope, 
                   colored_vessels, thalassemia],
            outputs=[results_text, results_plot, pdf_download]
        )
        
        # Professional Information Section
        gr.Markdown("## 🔬 Technical Information")
        
        # Add optional WandB tracking info
        if WANDB_AVAILABLE:
            gr.HTML("""<div class="info-box">
                <p style="margin: 0; color: #0c5460;">
                    <strong>📊 Analytics Tracking:</strong> This system includes optional Weights & Biases (WandB) 
                    integration for experiment tracking and analytics. When available, predictions and model 
                    performance are logged for continuous improvement and analysis.
                </p>
            </div>""")
        
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

if __name__ == "__main__":
    print("CardioPredict Pro - Loading models...")
    demo = create_interface()
    print("System ready - launching interface...")
    
    try:
        demo.launch(
            share=False,  # Hugging Face handles sharing
            inbrowser=False,  # Not needed on HF Spaces
            server_name="0.0.0.0",  # Required for HF Spaces
            server_port=7860,  # HF Spaces default port
            show_error=True,  # Show detailed errors
            quiet=False,  # Show startup logs
        )
    except KeyboardInterrupt:
        print("🛑 Shutting down gracefully...")
    finally:
        # Clean up WandB session if available
        if WANDB_AVAILABLE:
            try:
                finish_wandb_run()
                print("📊 WandB session finished")
            except Exception as e:
                print(f"⚠️ WandB cleanup warning: {e}")
