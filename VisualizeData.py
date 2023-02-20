import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

visualizeFlag = False

def setVisualizeFlag():
    global visualizeFlag 
    visualizeFlag = True

def plotCorrelationHeatmap(df):
    if not visualizeFlag:
        return
    corr = df.corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values)
    plt.show()
    return

def plotConfusionMatrix(test, predicted, title='confusion matrix'):
    if not visualizeFlag:
        return
    cm = confusion_matrix(test, predicted)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Greens', xticklabels=['lose', 'win'], yticklabels=['lose', 'win'])
    plt.xlabel('Predicted')
    plt.ylabel('Truth')
    plt.title(title)
    plt.show()
    return