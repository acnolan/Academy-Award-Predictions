import seaborn as sns
import matplotlib.pyplot as plt

def plotCorrelationHeatmap(df):
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()
    return