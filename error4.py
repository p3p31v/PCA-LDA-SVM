from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
mypath = Path().absolute() 
print(mypath)
## load the data
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
# Preparing input
X = df.iloc[:,:].values
y=dff
dfff=[]

for i in dff:
    dfff.append(i[2])
dfff=np.asarray(dfff)


enc = LabelEncoder()
label_encoder = enc.fit(dfff)
dfff = label_encoder.transform(dfff) + 1
print (dfff)
y=dfff
y = list(set(y))
y = np.asarray(y)
#calculating the number of clases
n_clases=y.shape
n_clases=n_clases[0]
print(n_clases) 

y=[1,1,1,2,4,2,3,3,4,3]
y = np.asarray(y)
n_clases=4
print (X.shape)
print (y.shape)
#cambiar, hacer un bucle con numeros en vez de letras tipo 1 2 etc:
list1=[]
list2=[]

for i in range(1,n_clases+1):
    list1.append(i)
    list2.append('tipo '+str(i))


label_dict = dict(zip(list1,list2))
print (label_dict)


# Applying LDA
sklearn_lda = LDA(n_components=2)
X_lda_sklearn = sklearn_lda.fit_transform(X, y)
const_colors = ["#C2272D", "#F8931F", "#FFFF01", "#009245", "#0193D9", "#0C04ED", "#612F90"]

# Plotting LDA
def plot_scikit_lda(X, title):
    if n_clases>2:
        s=1
    else:
        s=0
#cambiar:
    ax = plt.subplot(111)
    print (X.shape)
    for label,marker in zip(
        range(1,n_clases+1),range(3,n_clases+4)):

        plt.scatter(x=X[:,0][y == label],
                    y=X[:,s][y == label] * -1, # flip the figure
                    marker=marker,
                    
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



