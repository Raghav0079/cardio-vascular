#!/usr/bin/env python3
"""
CardioPredict Pro - WandB Project Initialization Script
Run this script to set up Weights & Biases tracking for the cardiovascular prediction project
"""

import sys
import os
from wandb_integration import initialize_wandb_project

def main():
    """Main initialization function"""
    print("🫀 CardioPredict Pro - WandB Integration Setup")
    print("=" * 50)
    
    # Check if wandb is available
    try:
        import wandb
        print("✅ WandB package detected")
    except ImportError:
        print("❌ WandB not installed. Install with: pip install wandb")
        return False
    
    # Check if user is logged in
    try:
        api = wandb.Api()
        user = api.viewer
        print(f"✅ Logged in as: {user.username if user else 'Unknown'}")
    except Exception:
        print("⚠️  Not logged in to WandB. Please run: wandb login")
        print("   Get your API key from: https://wandb.ai/authorize")
        
        # Ask if user wants to continue anyway
        response = input("\nContinue with initialization? (y/n): ")
        if response.lower() != 'y':
            return False
    
    print("\n🚀 Initializing CardioPredict Pro project...")
    
    # Initialize the project
    success = initialize_wandb_project()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Project initialization completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run your cardiovascular prediction app")
        print("2. Monitor experiments in WandB dashboard")
        print("3. Analyze model performance and patient trends")
        print("4. Export insights for medical research")
        
        print(f"\n🔗 Access your dashboard:")
        print(f"   https://wandb.ai/cardiopredict-pro")
        
    else:
        print("\n❌ Project initialization failed")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure you're logged in: wandb login")
        print("2. Check internet connection")
        print("3. Verify API key permissions")
        
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)