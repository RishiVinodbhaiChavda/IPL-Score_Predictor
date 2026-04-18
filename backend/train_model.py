"""Run this once to train and save the hybrid model."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ml.model_loader import train_and_save

if __name__ == "__main__":
    print("Training IPL Score Prediction Model...")
    xgb, nn, scaler = train_and_save()
    print("Training complete! Models saved to models/")
