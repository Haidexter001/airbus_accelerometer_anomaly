import pandas as pd
import numpy as np

class DataFrameProcessor:
    """
    A class to process and reshape DataFrames for machine learning workflows.
    """

    def __init__(self, step_size: int = 60, total_columns: int = 1024, progress_interval: int = 100, verbose: bool = True):
        """
        Initialize the processor with configurable parameters.

        Args:
            step_size (int): Step size for slicing ranges within rows. Default is 60.
            total_columns (int): Number of columns in the processed DataFrame. Default is 1024.
            progress_interval (int): Interval to print progress messages. Default is 100 rows.
            verbose (bool): Whether to print progress messages. Default is True.
        """
        self.step_size = step_size
        self.total_columns = total_columns
        self.progress_interval = progress_interval
        self.verbose = verbose

    def downsample_and_save(self, data: pd.DataFrame, output_path: str) -> None:
        """
        Processes a DataFrame by calculating averages over specified ranges and saves the result to an HDF5 file.

        Args:
            data (pd.DataFrame): Input DataFrame with the data to process.
            output_path (str): Path to save the processed DataFrame as an HDF5 file.
        """
        if data is None or data.empty:
            raise ValueError("Input data cannot be None or empty.")
        
        if self.step_size > data.shape[1]:
            raise ValueError("Step size cannot exceed the number of columns in the data.")
        
        # Initialize new DataFrame
        col_names = [i for i in range(self.total_columns)]
        new_df = pd.DataFrame(np.nan, index=[i for i in range(len(data))], columns=col_names)
        
        # Process data row by row
        for row in range(len(data)):
            start_point = 0
            finish_point = self.step_size
            
            for sub_col in range(self.total_columns):
                assigned_value = np.average(data.iloc[row][start_point:finish_point].values)
                new_df.iloc[row][sub_col] = assigned_value
                start_point += self.step_size
                finish_point += self.step_size
            
            if self.verbose and (row + 1) % self.progress_interval == 0:
                print(f"{row + 1} rows processed out of {len(data)}.")
        
        new_df.to_hdf(output_path, key='df', mode='w')

    def reshape_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Reshapes the input DataFrame for model input.

        Args:
            data (pd.DataFrame): Input DataFrame to reshape.

        Returns:
            np.ndarray: Reshaped data as a NumPy array.
        """
        if data is None or data.empty:
            raise ValueError("Input DataFrame cannot be None or empty.")
        reshape_data = data.values.reshape(data.shape[0], 1, data.shape[1])
        print(f"Data shape after reshaping: {reshape_data.shape}")
        return reshape_data
    
    def generate_test_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate labels for the test dataset.

        Args:
            test_data_path (str): File path to the HDF5 dataset containing the test data.

        Returns:
            np.ndarray: Array of labels extracted from the test dataset. Returned as array of booleans
        """
        try:
            df_abnormal = data
            df_abnormal['y'] = 1
            raw_data = df_abnormal.values
            labels = raw_data[:, -1]
        
            return labels.astype(bool)

        except FileNotFoundError:
            print(f"Error: File not found at path {data}")
            return np.array([])

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return np.array([])