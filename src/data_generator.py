import numpy as np
import pandas as pd
from typing import Tuple

class MetasurfaceDataGenerator:
    """
    Generate synthetic data for metasurface simulation.
    This is a placeholder class that generates random data.
    In a real application, this would interface with electromagnetic simulation software.
    """
    
    def __init__(self, num_samples: int = 1000):
        """
        Initialize the data generator.
        
        Args:
            num_samples (int): Number of samples to generate
        """
        self.num_samples = num_samples
        
    def generate_geometric_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random geometric parameters.
        
        Returns:
            Tuple of arrays containing width, height, and thickness
        """
        width = np.random.uniform(0.1, 2.0, self.num_samples)  # mm
        height = np.random.uniform(0.1, 2.0, self.num_samples)  # mm
        thickness = np.random.uniform(0.01, 0.5, self.num_samples)  # mm
        
        return width, height, thickness
    
    def generate_material_properties(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate random material properties.
        
        Returns:
            Tuple of arrays containing permittivity and permeability
        """
        permittivity = np.random.uniform(1.0, 10.0, self.num_samples)
        permeability = np.random.uniform(0.8, 1.2, self.num_samples)
        
        return permittivity, permeability
    
    def generate_operation_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate random operation parameters.
        
        Returns:
            Tuple of arrays containing frequency, theta, and phi
        """
        frequency = np.random.uniform(1.0, 20.0, self.num_samples)  # GHz
        theta = np.random.uniform(0, 90, self.num_samples)  # degrees
        phi = np.random.uniform(0, 360, self.num_samples)  # degrees
        
        return frequency, theta, phi
    
    def calculate_s_parameters(self, 
                             width: np.ndarray,
                             height: np.ndarray,
                             thickness: np.ndarray,
                             permittivity: np.ndarray,
                             permeability: np.ndarray,
                             frequency: np.ndarray,
                             theta: np.ndarray,
                             phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate S-parameters based on input parameters.
        This is a simplified model for demonstration purposes.
        
        Returns:
            Tuple of arrays containing S11 and S21 parameters
        """
        # Simplified model - in reality, this would be based on electromagnetic simulation
        S11 = np.abs(np.sin(frequency) * np.cos(theta/90) * 
                     np.exp(-thickness) * np.sqrt(permittivity))
        S21 = np.abs(np.cos(frequency) * np.sin(theta/90) * 
                     np.exp(-thickness) * np.sqrt(permeability))
        
        # Normalize to [0, 1]
        S11 = np.clip(S11, 0, 1)
        S21 = np.clip(S21, 0, 1)
        
        return S11, S21
    
    def generate_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset.
        
        Returns:
            DataFrame containing all parameters and S-parameters
        """
        # Generate parameters
        width, height, thickness = self.generate_geometric_parameters()
        permittivity, permeability = self.generate_material_properties()
        frequency, theta, phi = self.generate_operation_parameters()
        
        # Calculate S-parameters
        S11, S21 = self.calculate_s_parameters(
            width, height, thickness, permittivity, permeability, frequency, theta, phi
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'width': width,
            'height': height,
            'thickness': thickness,
            'permittivity': permittivity,
            'permeability': permeability,
            'frequency': frequency,
            'theta': theta,
            'phi': phi,
            'S11_magnitude': S11,
            'S21_magnitude': S21
        })
        
        return df

def main():
    """
    Generate and save synthetic dataset.
    """
    # Generate dataset
    generator = MetasurfaceDataGenerator(num_samples=10000)
    df = generator.generate_dataset()
    
    # Save to CSV
    df.to_csv('data/input.csv', index=False)
    print("Dataset generated and saved to data/input.csv")

if __name__ == "__main__":
    main() 