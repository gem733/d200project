import unittest
import pandas as pd
from src.load_data import load_data

class TestLoadCSVDataset(unittest.TestCase):
    
    file_path = "data/raw/constituencies_dataset.csv"

    def test_load_data(self):
        # Run the function
        df = load_data(self.file_path)
        
        # Check that a DataFrame is returned
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check that the DataFrame is not empty
        self.assertGreater(len(df), 0)
        
        # Optional: check that expected columns exist
        expected_columns = ["pubs", "margin", "swing"]
        for col in expected_columns:
            self.assertIn(col, df.columns)

if __name__ == "__main__":
    unittest.main()