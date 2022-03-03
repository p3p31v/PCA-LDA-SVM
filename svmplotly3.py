
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly import tools
import plotly.offline as py
from sklearn.feature_selection import SelectKBest
import statistics as statistics_formulas
from sklearn.feature_selection import chi2
#import plotly.graph_objs as go
#from plotly import tools
import plotly.tools as tls
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

## loading the data
file_name = 'peakmatrix_proteins.csv'
peak_matrix = pd.read_csv(file_name)

## Preparing data
df = peak_matrix.copy()
del df['Mass']

## Calculating  y as a vector with the classes and trasposing the sample matrix
df.tail()
dff=df.columns
df=df.T



## Preparing input
X = df.iloc[:,:].values



y=[1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]

X_new = SelectKBest(chi2, k=2).fit_transform(X, y)


y=np.asarray(y)

print (X_new)
print (y.shape)
k = list(set(y))
k = np.asarray(k)
n_clases=k.shape
n_clases=n_clases[0]

 
print (n_clases)
#X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)



X_new=np.asarray(X_new)

from sklearn.preprocessing import StandardScaler
st = StandardScaler()
X_new = st.fit_transform(X_new)

print (X_new.shape)
print(y.shape)
X=X_new
h = .02  # step size in the mesh


C = 1.0  # SVM regularization parameter
print (X)
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_ = np.arange(x_min, x_max, h)
y_ = np.arange(y_min, y_max, h)
xx, yy = np.meshgrid(x_, y_)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')



def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []
    
    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])
        
    return pl_colorscale

cmap = matplotlib_to_plotly(plt.cm.coolwarm, 5)

fig = tls.make_subplots(rows=2, cols=2,
                          print_grid=False,
                          subplot_titles=titles)



for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
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

for i in map(str, range(1, 5)):
    y = 'yaxis'+ i
    x = 'xaxis'+i
    fig['layout'][y].update(showticklabels=False, ticks='',
                            title="Sepal Length")
    fig['layout'][x].update(showticklabels=False, ticks='',
                            title="Sepal Width")

fig['layout'].update(height=700, showlegend=False)

py.plot(fig)