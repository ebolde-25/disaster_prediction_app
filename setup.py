#!/usr/bin/env python3
"""
Setup script for the African Disaster Prediction App
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️  Python 3.8 or higher is required")
        return False

    print(f"✅ Python {version.major}.{version.minor} is compatible")
    return True

def main():
    """Main setup function"""
    print("🌍 African Disaster Prediction App - Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        return 1

    # Install requirements
    if not install_requirements():
        return 1

    print("\n🎉 Setup completed successfully!")
    print("🚀 Run the application with: python run_app.py")
    print("📚 Or manually with: streamlit run app.py")

    return 0

if __name__ == "__main__":
    exit(main())
