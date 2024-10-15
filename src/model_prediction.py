import joblib
import sys
import pandas as pd
import os
from typing import Optional

class ModelPredictor:
    def __init__(self, model_path: str):
        """
        Initializes the predictor by loading the pre-trained model from the given path.
        
        Parameters:
        model_path (str): Path to the saved model file (e.g., 'model.joblib').
        """
        # Use the correct path to load model.joblib within a bundled executable
        if getattr(sys, 'frozen', False):
            # If the script is running as a bundled executable
            bundle_dir = sys._MEIPASS
            model_path = os.path.join(bundle_dir, 'model.joblib')
        else:
            # Running as a normal Python script
            model_path = 'model.joblib'
            
        self.model = joblib.load(model_path)  # Load the saved model

    def preprocess_input(self, file_path: str) -> pd.DataFrame:
        """
        Preprocesses the input CSV by:
        1. Reading the CSV file into a DataFrame.
        2. Dropping the 'target' column, if it exists, as it is not needed for predictions.
        
        Parameters:
        file_path (str): Path to the CSV file containing input data.
        
        Returns:
        pd.DataFrame: Preprocessed DataFrame containing only the feature columns (X).
        """
        data = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
        # Drop the target column if it exists, so we only have the feature columns
        if 'target' in data.columns:
            data = data.drop('target', axis=1)
        return data

    def make_predictions(self, input_csv: str) -> str:
        """
        Reads input data from a CSV, preprocesses it, makes predictions using the loaded model,
        appends the predictions to the input data, and saves the result as a new CSV file with
        the original filename suffixed by '_with_predictions'.
        
        Parameters:
        input_csv (str): Path to the input CSV file containing features.
        
        Returns:
        str: The path to the saved output CSV file with predictions.
        """
        # Preprocess the data
        data = self.preprocess_input(input_csv)
        
        # Make predictions using the preprocessed data
        predictions = self.model.predict(data)
        
        # Append the predictions to the original data
        data['predicted_target'] = predictions
        
        # Generate the new output filename by adding '_with_predictions' suffix
        base_name = os.path.splitext(input_csv)[0]  # Remove file extension
        output_csv = f"{base_name}_with_predictions.csv"
        
        # Save the new DataFrame with predictions to a CSV file
        data.to_csv(output_csv, index=False)
        
        # Return the output file path
        return output_csv

def main() -> None:
    """
    This part of the code only runs if the script is executed directly (not imported).
    It takes a command-line argument for the input CSV file, processes the data, 
    makes predictions, appends them to the input data, and saves the result as a new CSV.
    
    Usage (from command line):
    python predict_script.py path_to_input_csv
    
    Example:
    python predict_script.py test_data.csv
    """
    input_file: str = sys.argv[1]  # Get the first command-line argument (input CSV file)
    
    # Initialize the predictor with the path to the saved model
    predictor = ModelPredictor(model_path='model.joblib')
    
    # Make predictions and save the output CSV
    output_file: str = predictor.make_predictions(input_file)
    
    print(f"Predictions saved to: {output_file}")

# Ensure this script runs only when executed directly, not when imported as a module
if __name__ == '__main__':
    main()
