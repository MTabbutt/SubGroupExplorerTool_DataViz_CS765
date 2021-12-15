import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.mixture import GaussianMixture
import math

plt.rcParams.update({'legend.fontsize':'medium', 'font.size':14.0,
    'axes.titlesize':'large', 'axes.labelsize':'medium',
    'xtick.major.size':7,'xtick.minor.size':4,'xtick.major.pad':8,'xtick.minor.pad':8,'xtick.labelsize':'medium',
    'xtick.minor.width':1.0,'xtick.major.width':0.5,
    'ytick.major.size':7,'ytick.minor.size':4,'ytick.major.pad':8,'ytick.minor.pad':8,'ytick.labelsize':'medium',
    'ytick.minor.width':1.0,'ytick.major.width':1.0})


viridis_colors = ['#fde725', '#5ec962', '#21918c', '#3b528b', '#440154']

from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


class Explorer:
    
    """
    This class creates an explorer object from a dataset. 
    
    """
    
    def __init__(self, dataset_path):
        
        """
          skip_rows:     skip 2 rows: the first is an extra name column and the second is the data type column
          
        """
        self.dataset = pd.read_csv(dataset_path, skiprows=[0])
        self.correlation_matrix = self.getCorrelationMatrix()
        self.categories = pd.read_csv(dataset_path, nrows=1)
        #self.subgroupmatrix 
        
        
    def getDataFrame(self):
        
        return self.dataset
    
    
    
    def setDataFrame(self, df):
        
        self.dataset = df
        
        return self.dataset
    
    
      
        
##### Correlations #####


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
    
    
    
##### Distribution Statistics ##### 
    
    
    def getSummaryStats(self, col1, printStats=True):

        x = ~np.isnan(self.dataset[col1])
        z = self.dataset[col1] != np.Inf
        a = x & z 
        
        if len(self.dataset[col1][a]) == 0:
            return (-999, -999, -999, -999, -999)
        
        num = len(self.dataset[col1][a])
        ran = abs(max(self.dataset[col1][a]) - min(self.dataset[col1][a]))
        tmean = stats.tmean(self.dataset[col1][a])
        variance = stats.tvar(self.dataset[col1][a])
        skew = stats.skew(self.dataset[col1][a])
        
        if printStats:
            print("num: ", round(num, 6), ", range: ", round(ran, 6), ", mean: ", round(tmean, 6), 
                  ", variance: ", round(variance, 6), ", skew: ", round(skew, 6))
        
        return  (num, ran, tmean, variance, skew)
    
    
    def getSummaryMatrix(self, printStats=False):
        
        stats = ["counts", "range", "mean", "variance", "skew"]
        summary_matrix = []
        
        for col1 in self.dataset.columns:
            
            summary_matrix.append(self.getSummaryStats(col1, printStats=printStats))
        
        self.summarystats = pd.DataFrame(summary_matrix, columns=stats, index=self.dataset.columns)

        return self.summarystats
    
    
    
    def plotSummaryStats(self, col1, printStats=True):
        
        stats = ["counts", "range", "mean", "variance", "skew"]
        summaryStats = self.getSummaryStats(col1, printStats=printStats)
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 3]})
        
        # Counts and skew
        axs[0].hist(self.dataset[col1], bins=20, density=True, color=viridis_colors[2])
        self.dataset[col1].plot.kde(label="Kernel Density Estimator", color=viridis_colors[4], ax=axs[0])
        axs[0].legend(loc='best')
        axs[0].set_xlabel(col1)
        
        
        #Range, mean, variance:
        axs[1].boxplot(self.dataset[col1])
        axs[1].set_xlabel(col1)
        axs[1].set_ylabel("Values")
        axs[1].set_xlim(-1, 3)
        axs[1].annotate("Range: " + str(round(summaryStats[2], 3)), (.6, .9), xycoords="figure fraction")
        
        #plt.show()
        return
        
        
    def plotSummaryStats(self, col1, printStats=True):

        stats = ["counts", "range", "mean", "variance", "skew"]
        summaryStats = self.getSummaryStats(col1, printStats=printStats)

        x = self.dataset[col1]

        fig, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 2]})
        fig.tight_layout()

        # Counts and skew
        axs[0].hist(x, bins=20, density=True, color=viridis_colors[2], edgecolor="white", linewidth=.2)
        x.plot.kde(label="Kernel Density Estimator", color=viridis_colors[4], ax=axs[0], linewidth=2)
        axs[0].legend(loc='upper left')
        axs[0].set_xlim((min(x)-.1*max(x)), (max(x)+.1*max(x)))
        axs[0].set_xlabel(col1)
        axs[0].annotate("Skew: " + str(round(summaryStats[4], 3)), (.2, .8), xycoords="figure fraction")


        #Range, mean, variance:

        boxprops = dict(facecolor=viridis_colors[2], alpha=.4)
        meanprops = dict(linestyle='-', linewidth=2.5, color=viridis_colors[4])
        medianprops = dict(linestyle='--', linewidth=0, color='white')

        axs[1].boxplot(x, patch_artist=True, meanline=True, showmeans=True, 
                       meanprops=meanprops, medianprops=medianprops, boxprops=boxprops)
        axs[1].set_xlabel(col1)
        axs[1].set_ylabel("Values")

        axs[1].annotate("Range: " + str(round(summaryStats[1], 3)), (1.1, max(x)), xycoords="data")
        axs[1].annotate("Mean: " + str(round(summaryStats[2], 3)), (1.1, summaryStats[2]), xycoords="data")
        axs[1].annotate("StDev: " + str(round(summaryStats[3]**(.5), 3)), (1.1, min(x)+summaryStats[3]**(.5)), xycoords="data")

        return
    
    def plotSummaryStatsComparison(self, col1, col2, printStats=True):

        stats = ["counts", "range", "mean", "variance", "skew"]
        summaryStats1 = self.getSummaryStats(col1, printStats=printStats)
        summaryStats2 = self.getSummaryStats(col2, printStats=printStats)

        x1 = self.dataset[col1].dropna()
        x2 = self.dataset[col2].dropna()

        fig, axs = plt.subplots(1, 3, figsize=(17, 8), gridspec_kw={'width_ratios': [3, 3, 3]})
        fig.tight_layout()
        
        # Counts and skew #1
        axs[0].hist(x1, bins=20, density=True, color=viridis_colors[1], edgecolor="white", linewidth=.2, alpha=.5)
        x1.plot.kde(label="KDE: "+col1, color=viridis_colors[1], ax=axs[0], linewidth=2)
        
        axs[0].legend(loc='upper left')
        axs[0].set_xlim((min(x1)-.1*max(x1)), (max(x1)+.1*max(x1)))
        axs[0].set_xlabel(col1)
        axs[0].annotate("Skew: " + str(round(summaryStats1[4], 3)), (.25, .8), xycoords="figure fraction")
        
        
        # Counts and skew #2
        axs[1].hist(x2, bins=20, density=True, color=viridis_colors[2], edgecolor="white", linewidth=.2, alpha=.5)
        x2.plot.kde(label="KDE: "+col2, color=viridis_colors[2], ax=axs[1], linewidth=2)
        
        axs[1].legend(loc='upper left')
        axs[1].set_xlim((min(x2)-.1*max(x2)), (max(x2)+.1*max(x2)))
        axs[1].set_xlabel(col2)
        axs[1].annotate("Skew: " + str(round(summaryStats2[4], 3)), (.5, .8), xycoords="figure fraction")


        #Range, mean, variance:
        boxprops = dict(facecolor=viridis_colors[4], alpha=.4)
        meanprops = dict(linestyle='-', linewidth=2.5, color=viridis_colors[4])
        medianprops = dict(linestyle='--', linewidth=0, color='white')

        axs[2].boxplot([x1, x2], patch_artist=True, meanline=True, showmeans=True, 
                       meanprops=meanprops, medianprops=medianprops, boxprops=boxprops, positions=[1, 3])
        axs[2].set_xlabel(col1 + " and " + col2)
        axs[2].set_ylabel("Values")

        axs[2].annotate("Range: " + str(round(summaryStats1[1], 3)), (1.2, max(x1)), xycoords="data")
        axs[2].annotate("Mean: " + str(round(summaryStats1[2], 3)), (1.2, summaryStats1[2]), xycoords="data")
        axs[2].annotate("StDev: " + str(round(summaryStats1[3]**(.5), 3)), (1.2, min(x1)+summaryStats1[3]**(.5)*.75), xycoords="data")
        
        axs[2].annotate("Range: " + str(round(summaryStats2[1], 3)), (3.2, max(x2)), xycoords="data")
        axs[2].annotate("Mean: " + str(round(summaryStats2[2], 3)), (3.2, summaryStats2[2]), xycoords="data")
        axs[2].annotate("StDev: " + str(round(summaryStats2[3]**(.5), 3)), (3.2, min(x2)+summaryStats2[3]**(.5)*.75), xycoords="data")
        
        axs[2].set_xlim(0, 5)

        return
    
##### Clustering #####    


    def getClustering(self, clusterdata, ncomponents=7, reg_covar=.0001):
        
        if clusterdata == 'summary':
        
            self.gm = GaussianMixture(n_components=ncomponents, reg_covar=reg_covar).fit(self.getSummaryMatrix())
            arr = self.gm.predict(self.getSummaryMatrix())
            clusters = []

            for n in range(ncomponents):
                clusters.append(list(self.dataset.columns[arr == n]))

            self.clusters = clusters
    
            return self.gm.predict(self.getSummaryMatrix())

        return 
        
        
    def printClusters(self, clusterdata, ncomponents=7, reg_covar=.0001):
        
        self.getClustering(clusterdata, ncomponents=7, reg_covar=.0001)
            
        for cluster in self.clusters:
            print(cluster)
            print()
    
        return self.clusters
        
        
        
##### Similar Chocies #####           
        
        
    def selectSimilarColumn(self, col1, method, ncomponents=7, reg_covar=.0001):
    
        stats = ["counts", "range", "mean", "variance", "skew"]
        
        if method == "cluster":
            
            self.getClustering('summary', ncomponents=ncomponents, reg_covar=reg_covar)
          
            for cluster in self.clusters:
                if col1 in cluster:
                    return cluster
                
        elif method in stats:
            
            diff = 100000
            index = -999
            col1_data = self.summarystats[method][self.dataset.columns.get_loc(col1)]
            
            self.getSummaryMatrix()
            for i, item in enumerate(self.summarystats[method]): 
                if abs(col1_data - item) < diff and abs(col1_data - item) > .0001:
                    index = i
                    diff = abs(col1_data - item)
                
            print("most similar column: " + self.dataset.columns[index] + " diff: " + str(round(diff, 6)))
            
        return self.dataset.columns[index]
            
            
        
##### Comparing Subgroup Choices #####  


    def getCategoricalChoices(self):
        
        arr = []
        for i, cat in enumerate(self.categories):
            if cat[:11] == "Categorical":
                arr.append(self.dataset.columns[i])
      
                          
        return arr
    
   
        
    def getStatsSubGroup(self, df, col1, printStats=False):
        
        x = ~np.isnan(df[col1])
        z = df[col1] != np.Inf
        a = x & z 
        
        if len(df[col1][a]) == 0:
            return (-999, -999, -999, -999, -999)
        
        #print(df.info())
        
        num = len(df[col1][a])
        ran = abs(max(df[col1][a]) - min(df[col1][a]))
        tmean = stats.tmean(df[col1][a])
        variance = stats.tvar(df[col1][a])
        skew = stats.skew(df[col1][a])
        
        if printStats:
            print("num: ", round(num, 6), ", range: ", round(ran, 6), ", mean: ", round(tmean, 6), 
                  ", variance: ", round(variance, 6), ", skew: ", round(skew, 6))
        
        return  (num, ran, tmean, variance, skew)
    
    
    def getSubGroupsMatrix(self, df, categories, depVar1):
    
        matrix = []
        colortable = dict()

        for c, category in enumerate(categories):
            for i in np.unique(df[category].dropna()):
                df_sub = df[df[category] == i]
                arr = self.getStatsSubGroup(df_sub, depVar1, printStats=False)

                matrix.append([category+"_"+str(i)] + list(arr))
                colortable[category+"_"+str(i)] = cm.get_cmap('jet', len(categories))(1/(len(categories)+1)*(c+1))
               
        
        matrix_df = pd.DataFrame(np.array(matrix).transpose()[1:], columns=np.array(matrix).transpose()[0], 
                                 index=["counts", "range", "mean", "variance", "skew"])

        self.colortable = colortable
        
        self.subgroupmatrix = matrix_df
        
        return self.subgroupmatrix  
    
    
    def plotSubGroupMatrix(self, depVar1, categories, subcategories, style="overview"):
        
        from matplotlib.axes._axes import _log as matplotlib_axes_logger
        matplotlib_axes_logger.setLevel('ERROR')
        
        subgroupArr = np.array(self.subgroupmatrix).astype(np.float)

        self.getSummaryMatrix(printStats=False)
        base_arr = self.summarystats.loc[depVar1]
        
        
        if style == "overview":
            
            fy = .25*len(subcategories)+2
            
            if len(subcategories) < 16:
                fy = 6
                
            fig, axs = plt.subplots(1, 5, figsize=(18, fy), sharey=True)
            fig.tight_layout()

            for i, stat in enumerate(["counts", "range", "mean", "variance", "skew"]):

                axs[i].axvline(base_arr[stat], 0, len(list(self.subgroupmatrix.columns)), 
                               c='k', lw=2, ls="--", alpha=.6, zorder=1)
                
                for cat in categories:

                    for j, item in enumerate(subcategories):

                        axs[i].scatter(subgroupArr[i][j], j, c=self.colortable[item], zorder=4, s=75, edgecolors='gray')

                axs[i].grid(b=True, which='major', axis='y', c='gray', alpha=.4, lw=.5, zorder=1)
                axs[i].set_xlabel(stat)

            plt.yticks([i for i in range(len(list(subcategories)))], 
                       list(subcategories), fontsize=10)
            plt.suptitle("Summary Stats Changes for: "+depVar1+ " (baseline in black dashed)", y=1.02)

            
        plt.show()

        return
    
    
    
##### Subgroup comparison plots: #####    
    
    
    def getSubgroupCorrelation(self, df1, df2, col1, col2, plotting=True, alpha=.8):
        
        df = df1
        x, y = ~np.isnan(df[col1]), ~np.isnan(df[col2])
        z, w = df[col1] != np.Inf, df[col2] != np.Inf
        a = x & y & z & w
        
        corr1 = stats.pearsonr(df[col1][a], df[col2][a])
        
        df = df2
        x, y = ~np.isnan(df[col1]), ~np.isnan(df[col2])
        z, w = df[col1] != np.Inf, df[col2] != np.Inf
        a = x & y & z & w
        
        corr2 = stats.pearsonr(df[col1][a], df[col2][a])
        
        df = None
        
        if plotting:

            fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
            fig.tight_layout()
            
            axs[0].scatter(df1[col1], df1[col2], c=viridis_colors[3], alpha=alpha, label="r: "+str(round(corr1[0], 6)))
            axs[0].set_xlabel(col1)
            axs[0].set_ylabel(col2)
            axs[0].legend(loc="best")
            axs[0].title.set_text("Correlation of " + col1 + " and " + col2 + "Subgroup #1")
            
            axs[1].scatter(df2[col1], df2[col2], c=viridis_colors[3], alpha=alpha, label="r: "+str(round(corr2[0], 6)))
            axs[1].set_xlabel(col1)
            axs[1].legend(loc="best")
            axs[1].title.set_text("Correlation of " + col1 + " and " + col2 + "Subgroup #2")

        
        return (corr1, corr2)     
    
    
    
    
    def getSpecialPlot1(self, x, y1, y2, df1, df2, cmap="viridis", alpha=.5):


        cma = cm.get_cmap(cmap)
        clist1 = cma(df1[y2]/max(max(df1[y2]), max(df2[y2])))
        clist2 = cma(df2[y2]/max(max(df1[y2]), max(df2[y2])))

        fig, axs = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
        fig.tight_layout()

        each = max(1, math.floor(len(df1[y2])/80000))
        sc = axs[0].scatter(df1[x][::each], df1[y1][::each], c=clist1[::each], s=20, alpha=alpha, marker="^")
        axs[0].set_ylabel(y1)
        axs[0].set_xlabel(x)

        axs[1].scatter(df2[x][::each], df2[y1][::each], c=clist2[::each], s=20, alpha=alpha, marker="^")
        axs[1].set_xlabel(x)

        fig.colorbar(sc, ax=axs[:2], label=y2)


        plt.suptitle("Subgroup 1 vs Subgroup 2 Comparison", x=.42, y=1.02)
        
        
        return
    
    def getSpecialPlot2(self, x, y1, y2, df1, df2, labels, titleOn=True, titles=[], alpha=.85):
        
        matplotlib_axes_logger.setLevel('ERROR')

        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        fig.tight_layout()

        cat_vals = np.unique(np.array(df1[y2])) & np.unique(np.array(df2[y2]))

        x_1, x_2 = [], []
        y1_1, y2_1 = [], []

        for val in cat_vals:

            xx1, yy1 = [], []
            xx2, yy2 = [], []
            local_df1 = df1[df1[y2] == val]
            local_df2 = df2[df2[y2] == val]

            for item in np.unique(list(np.unique(local_df1[x])) + list(np.unique(local_df2[x]))):

                xx1.append(item)
                yy1.append(np.average(local_df1[local_df1[x] == item][y1]))
                xx2.append(item)
                yy2.append(np.average(local_df2[local_df2[x] == item][y1]))

            x_1.append(xx1)
            y1_1.append(yy1)
            x_2.append(xx2)
            y2_1.append(yy2)


        for nn in range(len(x_1)):
            axs[0].scatter(x_1[nn], y1_1[nn], s=20, alpha=alpha, label=labels[nn])
            axs[1].scatter(x_2[nn], y2_1[nn], s=20, alpha=alpha, label=labels[nn])

        axs[0].legend(loc='best')
        axs[1].legend(loc='best')
        axs[0].set_ylabel(y1)
        axs[0].set_xlabel(x)

        axs[1].set_xlabel(x)

        if titleOn:
            axs[0].set_title(titles[0])
            axs[1].set_title(titles[1])
    
    
        return 
    
    

    
    
    
    