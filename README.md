# 🫀 Cardiovascular Disease Prediction Interface

An interactive web interface for predicting cardiovascular disease risk using multiple machine learning models.

**Original Colab Notebook**: https://colab.research.google.com/drive/1Vb00nps377p8m_u_DYzMhfagoitD6A35?usp=sharing

## 🌟 Features

- **Interactive Web Interface**: Easy-to-use Gradio interface with intuitive input controls
- **Multiple ML Models**: Predictions from 4 different algorithms:
  - Logistic Regression (optimized)
  - Random Forest
  - Support Vector Machine (SVM)
  - Gradient Boosting
- **Real-time Predictions**: Instant risk assessment as you input health parameters
- **Visual Results**: Charts showing model predictions and confidence levels
- **Risk Assessment**: Clear recommendations based on model consensus
- **Shareable**: Generate public links to share the interface

## 🚀 Quick Start

### Option 1: Auto Setup (Recommended)
```bash
python launch.py
```

### Option 2: Manual Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the interface:
```bash
python gradio_interface.py
```

## 📊 Input Parameters

The interface accepts the following health parameters:

### Personal Information
- **Age**: Patient age (1-100 years)
- **Sex**: Gender (0: Female, 1: Male)

### Heart-Related Symptoms
- **Chest Pain Type**: 
  - 0: Typical Angina
  - 1: Atypical Angina
  - 2: Non-anginal Pain
  - 3: Asymptomatic
- **Exercise Induced Angina**: (0: No, 1: Yes)

### Clinical Measurements
- **Resting Blood Pressure**: mm Hg (80-250)
- **Cholesterol Level**: mg/dl (100-600)
- **Maximum Heart Rate**: Achieved during exercise (50-250)

### Lab Results
- **Fasting Blood Sugar**: > 120 mg/dl (0: No, 1: Yes)
- **Resting ECG Results**:
  - 0: Normal
  - 1: ST-T Wave Abnormality
  - 2: Left Ventricular Hypertrophy

### Additional Parameters
- **ST Depression**: Induced by exercise (0.0-10.0)
- **Slope**: Of peak exercise ST segment
  - 0: Upsloping
  - 1: Flat
  - 2: Downsloping
- **Number of Major Vessels**: Colored by fluoroscopy (0-3)
- **Thalassemia**:
  - 0: Normal
  - 1: Fixed Defect
  - 2: Reversible Defect
  - 3: Not described

## 📈 Output Interpretation

### Risk Levels
- **✅ LOW RISK**: Majority of models predict no heart disease
- **⚡ MODERATE RISK**: Some models indicate potential concerns
- **⚠️ HIGH RISK**: Multiple models detect potential heart disease

### Model Results
Each model provides:
- **Prediction**: Heart Disease Detected / No Heart Disease
- **Confidence**: Probability score for the prediction
- **Visualization**: Bar charts showing predictions and probabilities

## 🛠️ Technical Details

### Models Used
1. **Logistic Regression**: Optimized with C=1, liblinear solver
2. **Random Forest**: Ensemble of decision trees
3. **SVM**: Support Vector Machine with probability estimation
4. **Gradient Boosting**: Sequential ensemble learning

### Performance Metrics (from original analysis)
- **Logistic Regression**: 84% accuracy, 86% ROC AUC, 87% F1-score
- **Random Forest**: 77% accuracy, 88% ROC AUC, 81% F1-score
- **SVM**: 67% accuracy, 76% ROC AUC, 74% F1-score
- **Gradient Boosting**: 82% accuracy, 86% ROC AUC, 86% F1-score

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is for educational and research purposes
- **Not Medical Advice**: Results should not replace professional medical consultation
- **Consult Healthcare Professionals**: Always seek qualified medical advice for health concerns
- **Model Limitations**: Predictions are based on available data and may not account for all factors

## 📁 Files Structure

```
├── gradio_interface.py    # Main Gradio interface
├── launch.py             # Auto-setup and launch script
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── cardio_vascular.ipynb # Original analysis notebook
```

## 🔧 Customization

### Adding New Models
To add new models, modify the `load_and_prepare_models()` function in `gradio_interface.py`:

```python
# Add your new model
models['Your Model'] = YourModelClass()
models['Your Model'].fit(X_train, y_train)
```

### Modifying Interface
The Gradio interface can be customized by editing the `create_interface()` function:
- Change input components
- Modify layout
- Add new visualizations
- Update styling

## 🤝 Contributing

Feel free to contribute by:
- Adding new ML models
- Improving the interface design
- Adding new visualizations
- Enhancing the prediction logic
- Improving documentation

## 📞 Support

For issues or questions:
1. Check the console output for error messages
2. Ensure all dependencies are installed correctly
3. Verify that the CSV data file is accessible
4. Review the Gradio documentation for interface customization

---

**Made with ❤️ for cardiovascular health awareness**
