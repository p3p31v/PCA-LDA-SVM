
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly import tools
import plotly.offline as py

import statistics as statistics_formulas
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
#import plotly.graph_objs as go
#from plotly import tools
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

f = rfe.get_support(1) #the most important features
X = df[df.columns[f]] # final features`
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
svc1 = svm.SVC(kernel='linear', C=C).fit(X, y)
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_ = np.arange(x_min, x_max, h)
y_ = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(x_, y_)

# title for the plots
titles = ()



def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

cmap = matplotlib_to_plotly(plt.cm.coolwarm, 5)

fig = tls.make_subplots(rows=1, cols=2,
                          subplot_titles=titles)
print(fig.print_grid)
#he borrado nombres de rfe y demas y cambiado rows a 1
print (svc1)
for i, clf in enumerate((svc1,svc1)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    p1 = go.Contour(x=x_, y=y_, z=Z, 
                    colorscale=cmap,
                    showscale=False)
    fig.append_trace(p1, int(i/2+1), i%2+1)# me he inventao el int este para que no de error
    #pero no se si es correcto puesto que puede salir real en vez de entero creo

    # Plot also the training points
    p2 = go.Scatter(x=X[:, 0], y=X[:, 1], 
                    mode='markers',
                    marker=dict(color=y,
                                colorscale=cmap,
                                showscale=False,
                                line=dict(color='black', width=1))
                   )   
    fig.append_trace(p2, int(i/2+1), i%2+1)

for i in map(str, range(1, 3)):
    y = 'yaxis'+ i
    x = 'xaxis'+i
    fig['layout'][y].update(showticklabels=False, ticks='',
                            title="Feature 1")
    fig['layout'][x].update(showticklabels=False, ticks='',
                            title="Feature 2")

fig['layout'].update(height=700, showlegend=False)

py.plot(fig)