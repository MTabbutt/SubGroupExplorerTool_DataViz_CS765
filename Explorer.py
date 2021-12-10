import pandas as pd

class Explorer:
    
    """
    This class creates an explorer object from a dataset. 
    
    """
    
    def __init__(self, dataset_path, skip_rows=[1, 3]):
        
        """
          skip_rows:     skip 2 rows: the first is an extra name column and the second is the data type column
          
        """
        self.dataset = pd.read_csv(dataset_path, skiprows=skip_rows)

      
    
    
    def getDataFrame(self):
        return self.dataset