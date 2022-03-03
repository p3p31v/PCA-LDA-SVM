
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly import tools
import plotly.offline as py
import statistics as statistics_formulas
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import plotly.tools as tls
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

## applying rfe to linear model considering only the first two features
model = svm.SVC(kernel="linear")
rfe = RFE(model, 2)
## loading the data
file_name = 'peakmatrix_lipids4.csv'
peak_matrix = pd.read_csv(file_name)

## Preparing data
df = peak_matrix.copy()
del df['Mass']

## Calculating  y as a vector with the classes and trasposing the sample matrix
df.tail()
dff=df.columns
df=df.T
for ncol in df.columns: 
    df[ncol] = (df[ncol] - np.mean(df[ncol])) / statistics_formulas.stdev(df[ncol])


## Preparing input
X = df.iloc[:,:].values
y=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
rfe = rfe.fit(X, y)
f = rfe.get_support(1) 
X = df[df.columns[f]] 
X=np.asarray(X)
X_new=X

print (X_new)
## Scaling the input data
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X = st.fit_transform(X)
X_new=X
h = .02  # step size in the mesh


C = 1.0  # SVM regularization parameter
print (X)
clf = svm.SVC(kernel='linear', C=C).fit(X, y)


def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

cmap = matplotlib_to_plotly(plt.cm.coolwarm, 5)


# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_ = np.arange(x_min, x_max, h)
y_ = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(x_, y_)



Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)


trace1 = go.Heatmap(x=x_, y=y_, z=Z,
                    colorscale=cmap,
                    name='Marker',
                    showscale=False)

trace2 = go.Scatter(x=X[:, 0], y=X[:, 1], 
                    mode='markers',
                    name='Point',
                    marker=dict(color=y, 
                                colorscale=cmap, 
                                showscale=False,
                                line=dict(color='black', width=1)))

layout = go.Layout(title="SVM with Recursive Feature Elimination")
fig = go.Figure(data= [trace1, trace2], layout=layout)


py.plot(fig)