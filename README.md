# Deep Learning-Based Metasurface Coefficient Predictor

A deep learning model for predicting reflection (S11) and transmission (S21) coefficients of metasurfaces based on their geometric and material parameters. This project provides fast predictions without requiring full electromagnetic simulations.

## Project Structure 
/
│── data/ # Dataset files
│── src/
│ ├── utils.py # Utility functions for data processing and metrics
│ └── data_generator.py # Synthetic data generation
│── models/ # Saved model checkpoints
└── inference.py # Prediction script

## Features

- Synthetic data generation for metasurface parameters
- Data preprocessing and normalization utilities
- Model checkpoint management
- Performance metrics calculation (MSE, MAE, RMSE, R²)
- Command-line interface for predictions

## Usage

### 1. Generate Training Data

Generate a synthetic dataset with 10,000 samples:
bash
python src/data_generator.py


This creates `data/input.csv` containing:
- Geometric parameters (width, height, thickness)
- Material properties (permittivity, permeability)
- Operation parameters (frequency, theta, phi)
- S-parameters (S11_magnitude, S21_magnitude)

### 2. Train the Model

Train the model using the synthetic dataset:
bash
python src/train.py

### 3. Make Predictions

Make predictions using the trained model:
bash
python inference.py --input your_input.csv --output predictions.csv

Arguments:
- `--input`: Path to input CSV file (required)
- `--model_path`: Path to trained model (default: 'models/best_model.pth')
- `--output`: Path for output predictions (default: 'predictions.csv')

## Input Data Format

The input CSV should contain the following columns:
- width (mm)
- height (mm)
- thickness (mm)
- permittivity
- permeability
- frequency (GHz)
- theta (degrees)
- phi (degrees)

## Available Utilities (utils.py)

The utility module provides:
- `create_data_splits()`: Split data into train/validation/test sets
- `normalize_features()`: Normalize input features to [0,1] range
- `plot_training_history()`: Visualize training progress
- `calculate_metrics()`: Compute model performance metrics
- `save_model_checkpoint()`: Save model state and optimizer
- `load_model_checkpoint()`: Load saved model checkpoints

## Model Performance Metrics

The system calculates:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score (R²)


## License

This project is licensed under the MIT License. See the LICENSE file for details.


## Contact

For questions or feedback, please contact Elaheh Novinfard at elanvnfrd@gmail.com.










