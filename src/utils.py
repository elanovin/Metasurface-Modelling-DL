import numpy as np
import torch
from typing import Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

def create_data_splits(data_path: str, train_ratio=0.8, val_ratio=0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into train, validation, and test sets.
    
    Args:
        data_path (str): Path to the input CSV file
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        
    Returns:
        Tuple containing train, validation, and test indices
    """
    data = np.load(data_path)
    indices = np.random.permutation(len(data))
    
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Normalize features to [0, 1] range.
    
    Args:
        features (np.ndarray): Input features
        
    Returns:
        Tuple containing normalized features and the scaler object
    """
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    return normalized_features, scaler

def plot_training_history(history: Dict[str, list]) -> None:
    """
    Plot training and validation loss history.
    
    Args:
        history (Dict[str, list]): Dictionary containing training history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various performance metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dictionary containing various metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    # R-squared score
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def save_model_checkpoint(model: torch.nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        path (str): Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_model_checkpoint(model: torch.nn.Module,
                         optimizer: torch.optim.Optimizer,
                         path: str) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """
    Load model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load into
        optimizer (torch.optim.Optimizer): The optimizer to load into
        path (str): Path to the checkpoint
        
    Returns:
        Tuple containing loaded model, optimizer, epoch, and loss
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    return model, optimizer, epoch, loss 