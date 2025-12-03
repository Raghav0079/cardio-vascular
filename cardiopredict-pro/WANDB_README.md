# WandB Integration for CardioPredict Pro

This directory contains the Weights & Biases integration for comprehensive experiment tracking, model monitoring, and analytics for the CardioPredict Pro cardiovascular risk assessment system.

## 🚀 Quick Start

### 1. Initialize WandB Project
```bash
python init_wandb.py
```

### 2. Run with Tracking (Optional Integration)
If you want to add tracking to the main app, import the integration:

```python
from wandb_integration import (
    init_wandb_tracking,
    log_training_data, 
    log_prediction_data,
    create_wandb_dashboard,
    finish_wandb_run
)
```

## 📊 What Gets Tracked

### Model Training Phase
- Dataset statistics (size, features, class balance)
- Model configurations and initialization
- Training/validation split information
- Feature engineering pipeline

### Prediction Phase  
- Patient demographics (anonymized)
- Individual model predictions and probabilities
- Ensemble consensus and confidence scores
- Risk factor analysis
- Clinical parameter patterns

### System Metrics
- Interface performance
- PDF generation statistics  
- Session duration and usage patterns
- Model agreement patterns

## 🎯 Dashboard Features

### Real-time Monitoring
- **Model Performance**: Track individual model predictions and confidence
- **Ensemble Analytics**: Monitor voting patterns and consensus levels
- **Patient Insights**: Analyze risk factor distributions and trends
- **System Health**: Track interface usage and performance metrics

### Analytics Views
- **Risk Factor Heatmaps**: Visualize correlation between patient parameters and predictions
- **Model Comparison**: Compare performance across the 4-model ensemble
- **Confidence Distribution**: Analyze prediction confidence patterns
- **Temporal Trends**: Track prediction patterns over time

## 🔧 Configuration

### Project Configuration
```python
config = {
    "models": ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"],
    "features": 13 clinical parameters,
    "ensemble_method": "majority_voting",
    "output_format": "professional_pdf",
    "deployment": "huggingface_spaces"
}
```

### Privacy & Compliance
- All patient data is anonymized before logging
- No personally identifiable information is stored
- HIPAA-conscious data handling
- Educational/research use disclaimer

## 📈 Key Metrics Tracked

### Patient Level
- `prediction/patient_age` - Age category
- `prediction/resting_bp` - Blood pressure readings
- `prediction/cholesterol` - Cholesterol levels  
- `analysis/high_risk_factors_count` - Number of risk factors

### Model Level  
- `models/{model}_prediction` - Binary prediction (0/1)
- `models/{model}_risk_probability` - Risk probability (0-1)
- `models/{model}_confidence` - Confidence score (0-1)

### Ensemble Level
- `ensemble/positive_predictions` - Number of models predicting risk
- `ensemble/consensus_ratio` - Agreement percentage
- `ensemble/risk_category` - Final risk level (0-2)

### System Level
- `system/total_predictions` - Total predictions made
- `system/pdf_generation_enabled` - PDF functionality status
- `system/interface_version` - Version tracking

## 🌐 Dashboard Access

After initialization, access your dashboard at:
- **Project Dashboard**: https://wandb.ai/[username]/cardiopredict-pro
- **Run Details**: Individual session tracking with detailed metrics
- **Comparative Analysis**: Multi-run comparison and trends

## 🔗 Integration Examples

### Basic Integration
```python
# Initialize tracking
init_wandb_tracking()

# Log training data  
log_training_data(models, dataset)

# Log predictions
log_prediction_data(patient_data, results)

# Create dashboard
create_wandb_dashboard()

# Finish session
finish_wandb_run()
```

### Advanced Analytics
```python
# Custom metrics
wandb.log({
    "custom/high_risk_percentage": risk_percentage,
    "custom/model_agreement_score": agreement_score
})

# Interactive plots
wandb.log({
    "charts/risk_distribution": wandb.plot.histogram(risk_data)
})
```

## 🛠️ Troubleshooting

### Common Issues
1. **Login Issues**: Run `wandb login` and get API key from https://wandb.ai/authorize
2. **Network Issues**: Check internet connection for cloud sync
3. **Permission Issues**: Ensure write access to project directory

### Debug Mode
```python
import wandb
wandb.init(mode="offline")  # Work offline
wandb.init(mode="disabled")  # Disable tracking
```

## 📚 Resources

- **WandB Documentation**: https://docs.wandb.ai/
- **API Reference**: https://docs.wandb.ai/ref/python
- **Best Practices**: https://docs.wandb.ai/guides/track/best-practices
- **Medical AI Tracking**: https://wandb.ai/site/solutions/healthcare

---

*CardioPredict Pro v2.0 - AI-Powered Cardiovascular Risk Assessment with Advanced Analytics*