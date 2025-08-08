#!/usr/bin/env python3
"""
Standalone training script for the disaster prediction model
Run this to train the model separately from the Streamlit app
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_preprocessor import DisasterDataPreprocessor
from models.model_trainer import DisasterPredictor

def main():
    """Main training function"""
    print("🚀 Starting disaster prediction model training...")

    # Initialize components
    preprocessor = DisasterDataPreprocessor()
    trainer = DisasterPredictor()

    try:
        # Process data
        print("📊 Processing disaster data...")
        X, y = preprocessor.process_full_pipeline()

        # Train model
        print("🤖 Training XGBoost model...")
        metrics = trainer.train_model(X, y)

        # Save model
        print("💾 Saving trained model...")
        trainer.save_model()

        print("✅ Training completed successfully!")
        print(f"📈 Model Performance:")
        print(f"   R² Score: {metrics.get('r2_score', 0):.4f}")
        print(f"   MSE: {metrics.get('mse', 0):,.0f}")
        print(f"   MAE: {metrics.get('mae', 0):,.0f}")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
