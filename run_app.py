#!/usr/bin/env python3
"""
Simple launcher for the disaster prediction app
"""
import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    try:
        # Change to app directory
        app_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(app_dir)

        # Launch streamlit
        print("🚀 Launching African Disaster Prediction App...")
        print("📱 The app will open in your default browser")
        print("🌐 URL: http://localhost:8501")
        print("⭐ Press Ctrl+C to stop the application")
        print("-" * 50)

        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8501"
        ])

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("💡 Try running manually: streamlit run app.py")

if __name__ == "__main__":
    main()
