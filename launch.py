"""
🫀 Cardiovascular Disease Prediction - Gradio Interface
=====================================================

Quick setup and launch script for the heart disease prediction interface.

Usage:
1. Install dependencies: pip install -r requirements.txt
2. Run this script: python launch.py
3. Open the provided URL in your browser

The interface will automatically:
- Load and train all ML models (Logistic Regression, Random Forest, SVM, Gradient Boosting)
- Create an interactive web interface
- Provide real-time predictions with visualizations
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies. Please install manually:")
        print("pip install gradio pandas numpy scikit-learn matplotlib seaborn")
        return False
    return True

def launch_interface():
    """Launch the Gradio interface"""
    print("🚀 Launching Cardiovascular Disease Prediction Interface...")
    print("🌐 The interface will open in your default browser")
    print("📱 A shareable link will also be provided for remote access")
    
    try:
        # Import and run the interface
        from gradio_interface import create_interface
        
        demo = create_interface()
        demo.launch(
            share=True,  # Create shareable link
            inbrowser=True,  # Open in browser automatically
            server_name="0.0.0.0",  # Allow external access
            show_tips=True
        )
    except ImportError as e:
        print(f"❌ Error importing interface: {e}")
        print("Make sure gradio_interface.py is in the same directory")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")

if __name__ == "__main__":
    print("🫀 Cardiovascular Disease Prediction System")
    print("=" * 50)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found in current directory")
        sys.exit(1)
    
    # Install dependencies
    if install_requirements():
        print("\n" + "=" * 50)
        launch_interface()
    else:
        print("❌ Setup failed. Please install dependencies manually and try again.")