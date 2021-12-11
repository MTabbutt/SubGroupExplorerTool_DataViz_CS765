import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


viridis_colors = ['#fde725', '#5ec962', '#21918c', '#3b528b', '#440154']


class Explorer:
    
    """
    This class creates an explorer object from a dataset. 
    
    """
    
    def __init__(self, dataset_path, skip_rows=[1, 3]):
        
        """
          skip_rows:     skip 2 rows: the first is an extra name column and the second is the data type column
          
        """
        self.dataset = pd.read_csv(dataset_path, skiprows=skip_rows)
        self.correlation_matrix = self.getCorrelationMatrix()
        
        
        
    def getDataFrame(self):
        
        return self.dataset
    
    
       
        
##### Correlations: 


    def getCorrelation(self, col1, col2):

        x, y = ~np.isnan(self.dataset[col1]), ~np.isnan(self.dataset[col2])
        z, w = self.dataset[col1] != np.Inf, self.dataset[col2] != np.Inf
        a = x & y & z & w
        
        return stats.pearsonr(self.dataset[col1][a], self.dataset[col2][a]) 

    
    
    def getCorrelationMatrix(self):
        
        correlation_matrix = []
        
        for col1 in self.dataset:
            array = []
            
            for col2 in self.dataset:
                array.append(self.getCorrelation(col1, col2)[0])
                
            correlation_matrix.append(array)
        
        return pd.DataFrame(correlation_matrix, columns=self.dataset.columns, index=self.dataset.columns)
    
    def getMaxCorrelation(self, threshold=False):
        """
        Returns the maximum and minimum values if no threshold set, otherwise returns the max 
        and min above/below that threshold magnitude. 
        """
        
        if threshold:
            maxVal, minVal = [], []
            maxcol1, maxcal2, mincol1, mincol2 = [], [], [], []
            
            for col in self.correlation_matrix:
                
                for i, val in enumerate(self.correlation_matrix[col]):
                    
                    if val > threshold and val < .999:
                        maxVal.append(val)
                        maxcol1.append(self.correlation_matrix.columns[i])
                        maxcal2.append(col)
                        
                    if val < -threshold and val > -.999:
                        minVal.append(val)
                        mincol1.append(self.correlation_matrix.columns[i])
                        mincol2.append(col)
                        
            return [(maxcol1, maxcal2, maxVal), (mincol1, mincol2, minVal)]
        
        else:
            maxVal, minVal = 0, 0
            maxcol1, maxcal2, mincol1, mincol2 = "", "", "", ""
            
            for col in self.correlation_matrix:
                
                for i, val in enumerate(self.correlation_matrix[col]):
                    
                    if val > maxVal and val < .999:
                        maxVal = val
                        maxcol1 = self.correlation_matrix.columns[i]
                        maxcal2 = col
                        
                    if val < minVal and val > -.999:
                        minVal = val
                        mincol1 = self.correlation_matrix.columns[i]
                        mincol2 = col
                    
        
            return [(maxcol1, maxcal2, maxVal), (mincol1, mincol2, minVal)]  
        
        return [(), ()]    
    

    def plotCorrelation(self, col1, col2, alpha=.75):
        
        x, y = ~np.isnan(self.dataset[col1]), ~np.isnan(self.dataset[col2])
        z, w = self.dataset[col1] != np.Inf, self.dataset[col2] != np.Inf
        a = x & y & z & w
        
        print("Pearson's correlation coefficent: r = " + str(self.getCorrelation(col1, col2)[0].round(6)))
        
        plt.figure(figsize=(11, 7))

        plt.scatter(self.dataset[col1][a], self.dataset[col2][a], c=viridis_colors[3], alpha=alpha)
        plt.xlabel(col1, fontsize=16)
        plt.ylabel(col2, fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.title("Correlation of " + col1 + " and " + col2, size=18)
        
        return 
    
    def printCorrelationMatrix(self, correlation_matrix):
        
        pd.DataFrame(correlation_matrix)
        
        return 
    
    
    
##### Distribution Statistics:   
    
    
    
    
    
    
    
    
    
    
    