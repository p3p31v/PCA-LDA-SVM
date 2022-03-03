import sklearn
sklearn.__version__


import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names



lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components


colors = ['navy', 'turquoise', 'darkorange']



fig = tools.make_subplots(rows=1, cols=1,
                          subplot_titles=()
                         )


    
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    lda = go.Scatter(x=X_r2[y == i, 0], 
                     y=X_r2[y == i, 1],
                     showlegend=False,
                     mode='markers',
                     marker=dict(color=color),
                     name=target_name
                     )
    
    fig.append_trace(lda, 1, 1)
    

    
py.plot(fig)