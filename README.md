# 🫀 Cardiovascular Disease Prediction System

A comprehensive machine learning project for cardiovascular disease prediction featuring:
- **Complete ML Pipeline**: Data analysis, model training, and evaluation with 4 algorithms
- **Interactive Gradio Interface**: Real-time web application for risk assessment
- **Multiple Model Comparison**: Logistic Regression, Random Forest, SVM, and Gradient Boosting

**🔗 Original Analysis**: [Google Colab Notebook](https://colab.research.google.com/drive/1Vb00nps377p8m_u_DYzMhfagoitD6A35?usp=sharing)

## 🌟 Features

### 🤖 Machine Learning Pipeline
- **4 Trained Models**: Logistic Regression (87% F1), Random Forest (88% AUC), SVM, Gradient Boosting
- **Comprehensive Analysis**: EDA, feature correlation, model comparison in Jupyter notebook
- **Hyperparameter Tuning**: Optimized Logistic Regression with GridSearchCV
- **Performance Metrics**: Accuracy, ROC-AUC, F1-score, and confusion matrices

### 🌐 Interactive Web Interface
- **Real-time Predictions**: Instant risk assessment via Gradio web app
- **Multi-model Consensus**: Combines predictions from all 4 algorithms
- **Visual Analytics**: Interactive charts and confidence visualizations
- **Risk Stratification**: Clear Low/Moderate/High risk categories
- **Medical Context**: Educational disclaimers and healthcare recommendations

### 📊 Data Analysis
- **Exploratory Data Analysis**: Comprehensive visualization of cardiovascular indicators
- **Feature Engineering**: Analysis of age, cholesterol, blood pressure, heart rate patterns
- **Statistical Insights**: Correlation matrices and distribution analysis

## 🚀 Quick Start

### 🌐 Launch Web Interface

1. **Create local directory and files:**
```powershell
mkdir cardio_interface
cd cardio_interface
```

2. **Install dependencies:**
```bash
pip install gradio pandas numpy scikit-learn matplotlib seaborn
```

3. **Download and run the interface:**
   - Copy `gradio_interface.py` to your local directory
   - Run: `python gradio_interface.py`
   - Open: http://localhost:7860

### 📊 Analyze Original Data

- **Jupyter Notebook**: Open `cardio_vascular.ipynb` for complete analysis
- **Dataset**: `Cardio_vascular.csv` (downloadable via the notebook)

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

## 🖥️ Using the Web Interface

### 🎛️ Input Parameters
The interface provides intuitive controls for:

**👤 Personal Information**
- Age slider (1-100 years)
- Sex selection (Male/Female)

**💓 Cardiovascular Symptoms** 
- Chest pain type (4 categories)
- Exercise-induced angina (Yes/No)

**🩺 Clinical Measurements**
- Resting blood pressure (80-250 mmHg)
- Cholesterol level (100-600 mg/dl)
- Maximum heart rate (50-250 bpm)

**🔬 Laboratory Results**
- Fasting blood sugar levels
- ECG abnormalities
- Additional cardiac parameters

### 📊 Results Dashboard

**🎯 Risk Assessment**
- **✅ LOW RISK**: 0-1 models detect disease (Green)
- **⚡ MODERATE RISK**: 2 models detect disease (Orange) 
- **⚠️ HIGH RISK**: 3+ models detect disease (Red)

**📈 Model Predictions**
- Individual confidence scores for each algorithm
- Comparative bar charts showing prediction consensus
- Probability percentages for heart disease likelihood

**💡 Medical Recommendations**
- Personalized advice based on risk level
- Healthcare consultation guidance
- Lifestyle recommendations

## 🛠️ Technical Details

### Models Used
1. **Logistic Regression**: Optimized with C=1, liblinear solver
2. **Random Forest**: Ensemble of decision trees
3. **SVM**: Support Vector Machine with probability estimation
4. **Gradient Boosting**: Sequential ensemble learning

### 📈 Model Performance (Test Set Results)

| Model | Accuracy | ROC-AUC | F1-Score | Best For |
|-------|----------|---------|-----------|----------|
| **Logistic Regression** | 84% | 86% | **87%** | **Overall Balance** |
| **Random Forest** | 77% | **88%** | 81% | **Discrimination** |
| **SVM** | 67% | 76% | 74% | Feature Learning |
| **Gradient Boosting** | 82% | 86% | 86% | Complex Patterns |

**🏆 Best Model**: Logistic Regression (highest F1-score for medical applications)
**🎯 Key Insight**: High F1-score crucial for minimizing false negatives in heart disease detection

## ⚠️ Important Disclaimers

- **Educational Purpose Only**: This tool is for educational and research purposes
- **Not Medical Advice**: Results should not replace professional medical consultation
- **Consult Healthcare Professionals**: Always seek qualified medical advice for health concerns
- **Model Limitations**: Predictions are based on available data and may not account for all factors

## 📁 Project Structure

```
cardio-vascular/
├── 📊 DATA & ANALYSIS
│   ├── Cardio_vascular.csv      # Heart disease dataset
│   └── cardio_vascular.ipynb    # Complete ML analysis notebook
│
├── 🌐 WEB INTERFACE 
│   ├── gradio_interface.py      # Interactive Gradio web app
│   ├── requirements.txt         # Python dependencies
│   └── launch.py                # Setup and launch script
│
├── 📖 DOCUMENTATION
│   └── README.md                # This comprehensive guide
│
└── 🔧 CONFIGURATION
    └── .gitignore               # Git ignore rules
```

### 💾 Local Setup Files
When you run the interface locally, these files are created:
- `C:/Users/{username}/cardio_interface/gradio_interface.py`
- `C:/Users/{username}/cardio_interface/requirements.txt`

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

## 🔬 Research & Development

### 📈 Model Insights
- **Logistic Regression**: Best for medical interpretability and balanced performance
- **Random Forest**: Highest discrimination ability (ROC-AUC = 88%)
- **Hyperparameter Tuning**: GridSearchCV optimization improved model robustness
- **Feature Importance**: Age, chest pain type, and cholesterol are key predictors

### 🛠️ Technical Implementation
- **Backend**: scikit-learn for ML, pandas for data processing
- **Frontend**: Gradio for interactive web interface
- **Deployment**: Local hosting with optional public sharing
- **Visualization**: matplotlib + seaborn for statistical charts

## 🤝 Contributing

**🔧 Enhancement Ideas:**
- Add feature importance visualizations
- Implement additional ML models (XGBoost, Neural Networks)
- Create mobile-responsive design
- Add patient data export functionality
- Integrate with electronic health records (EHR)

**📊 Research Extensions:**
- Cross-validation analysis
- External dataset validation
- Explainable AI (SHAP values)
- Uncertainty quantification

## 🏥 Medical Disclaimer

**⚠️ IMPORTANT**: This tool is for **educational and research purposes only**
- Results should **never replace professional medical diagnosis**
- Always consult qualified healthcare professionals for medical concerns
- The models are trained on limited data and may not account for all factors
- This tool does not constitute medical advice or treatment recommendations

## 📞 Support & Documentation

- **📧 Issues**: Report bugs or request features via GitHub Issues
- **📖 Documentation**: See inline code comments and docstrings
- **🎓 Learning**: Study the Jupyter notebook for ML methodology
- **🌐 Gradio Docs**: [Official Gradio Documentation](https://gradio.app/docs/)

---

**🫀 Made with ❤️ for advancing cardiovascular health through AI**

*Contributing to better heart health outcomes through accessible machine learning tools.*
