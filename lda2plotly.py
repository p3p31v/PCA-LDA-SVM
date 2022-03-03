import sklearn
import statistics as statistics_formulas
import numpy as np
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

## loading the data
file_name = 'EFSvsEFVanBvsEFVanA_nopeak_mt3.csv'
peak_matrix = pd.read_csv(file_name)

## Preparing data
df = peak_matrix.copy()
del df['Mass']

## Calculating  y as a vector with the classes and trasposing the sample matrix
df.tail()
dff=df.columns
df=df.T

## Scaling data
for ncol in df.columns: 
    df[ncol] = (df[ncol] - np.mean(df[ncol])) / statistics_formulas.stdev(df[ncol])

## Preparing input
X = df.iloc[:,:].values


y=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

## Calculating the number of classes
y=np.asarray(y)
k = list(set(y))
k = np.asarray(k)
n_clases=k.shape
n_clases=n_clases[0]

# Making a dictionary to associate classes with points
list1=[]
list2=[]
for i in range(1,n_clases+1):
    list1.append(i)
    list2.append('Type '+str(i))

label_dict = dict(zip(list1,list2))
target_names=[]
for i in label_dict:
    target_names.append(label_dict[i])

## Applying LDA
lda = LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto',n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
fig = tools.make_subplots(rows=1, cols=1,
                          subplot_titles=()
                         )
## Plotting data
    
for i, target_name in zip(k, target_names):
    lda = go.Scatter(x=X_r2[y == i, 0], 
                     y=X_r2[y == i, 1],
                     showlegend=False,
                     mode='markers',
                    name=target_name
                     )
    
    fig.append_trace(lda, 1, 1)
    

    
py.plot(fig)