import torch
import pandas as pd
import numpy as np
from src.model import MetasurfacePredictor
from src.utils import normalize_features
import argparse
from pathlib import Path

def load_model(model_path: str) -> MetasurfacePredictor:
    """
    Load the trained model.
    
    Args:
        model_path (str): Path to the model checkpoint
        
    Returns:
        Loaded model
    """
    model = MetasurfacePredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(model: MetasurfacePredictor, input_data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions using the loaded model.
    
    Args:
        model (MetasurfacePredictor): Loaded model
        input_data (pd.DataFrame): Input data for prediction
        
    Returns:
        Numpy array of predictions
    """
    # Normalize input features
    normalized_features, _ = normalize_features(input_data.values)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(normalized_features)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(features_tensor)
    
    return predictions.numpy()

def main(args):
    # Load input data
    input_data = pd.read_csv(args.input)
    
    # Load model
    model = load_model(args.model_path)
    
    # Make predictions
    predictions = predict(model, input_data)
    
    # Create output DataFrame
    output_df = pd.DataFrame(predictions, columns=['S11_magnitude', 'S21_magnitude'])
    
    # Save predictions
    output_path = Path(args.output)
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth', 
                        help='Path to trained model')
    parser.add_argument('--output', type=str, default='predictions.csv',
                        help='Path to save predictions')
    
    args = parser.parse_args()
    main(args) 