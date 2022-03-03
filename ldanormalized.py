from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
import statistics as statistics_formulas

## loading the data
#file_name = 'LipidsPeakMatrix.csv'
file_name = 'peakmatrix_proteins.csv'
#file_name ='peakmatrix_lipids4.csv'
peak_matrix = pd.read_csv(file_name)

## Prepare data
df = peak_matrix.copy()
del df['Mass']

## calculating  y as a vector with the clases and trasposing the sample matrix
df.tail()
dff=df.columns
df=df.T
## normalizing data
for ncol in df.columns: 
    df[ncol] = (df[ncol] - np.mean(df[ncol])) / statistics_formulas.stdev(df[ncol])

# Preparing input
X = df.iloc[:,:].values
y=dff

## calculating the number of classes for a file with a format like LipidsPeakMatrix.csv
dfff=[]
for i in dff:
    dfff.append(i[2])
dfff=np.asarray(dfff)
enc = LabelEncoder()
label_encoder = enc.fit(dfff)
dfff = label_encoder.transform(dfff) + 1
y=dfff
y = list(set(y))
y = np.asarray(y)
n_clases=y.shape
n_clases=n_clases[0]

y=dfff
y=[1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
#y=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
n_clases=4
y=np.asarray(y)
# Making a dictionary to asociate clases with points
list1=[]
list2=[]
for i in range(1,n_clases+1):
    list1.append(i)
    list2.append('Type '+str(i))

label_dict = dict(zip(list1,list2))
print (label_dict)

# Applying LDA
sklearn_lda = LDA(solver='eigen',shrinkage='auto',n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)

# Plotting LDA, we plot in one dimension for n_clases=2
def plot_scikit_lda(X, title):
    if n_clases>2:
        s=1
    else:
        s=0
    ax = plt.subplot(111)
    print (X.shape)
    for label,marker in zip(
        range(1,n_clases+1),range(3,n_clases+4)):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,s][y == label] * -1, # flip the figure
                    marker=marker,
                    
                    alpha=0.5,
                    label=label_dict[label])
    if s==1:
        plt.xlabel('LD1')
        plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout
    plt.show()

plot_scikit_lda(X_lda_sklearn, title='LDA')