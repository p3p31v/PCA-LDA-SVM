from matplotlib import pyplot as plt



import numpy as np
import pandas as pd

from pathlib import Path 
mypath = Path().absolute() 
print(mypath)

file_name = 'LipidsPeakMatrix.csv'
peak_matrix = pd.read_csv(file_name)

## Prepare data
df = peak_matrix.copy()
data_mass = df['Mass'].copy().tolist()
del df['Mass']
df.dropna(how="all", inplace=True) # to drop the empty line at file-end

df.tail()
dff=df.columns
df=df.T

from sklearn.preprocessing import LabelEncoder
X = df.iloc[:,:].values
y=dff
dfff=[]
for i in dff:
    if i=='L1a1' or i=='L1a2' or i=='L1a3' or i== 'L1a4' or i== 'L1a5':
        dfff.append(1)
    else:
        dfff.append(2)

y=dfff


y=[1,1,1,2,2,2,3,3,3,3]
y = np.asarray(y)
print (X.shape)
print (y.shape)

label_dict = {1: 'tipo a', 2: 'tipo b',3: 'tipo c'}
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)



def plot_scikit_lda(X, title):

    ax = plt.subplot(111)
    print (X.shape)
    for label,marker,color in zip(
        range(1,4),('^', 's','o'),('blue', 'red','green')):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    color=color,
                    alpha=0.5,
                    label=label_dict[label])

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

plot_scikit_lda(X_lda_sklearn, title='Default LDA via scikit-learn')


