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

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda = LinearDiscriminantAnalysis(n_components=2)
X_r2 = lda.fit(X, y).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

colors = ['navy', 'turquoise', 'darkorange']



fig = tools.make_subplots(rows=1, cols=2,
                          subplot_titles=('PCA of IRIS dataset',
                                          'LDA of IRIS dataset')
                         )

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    pca = go.Scatter(x=X_r[y == i, 0], 
                     y=X_r[y == i, 1], 
                     mode='markers',
                     marker=dict(color=color),
                     name=target_name
                    )
    
    fig.append_trace(pca, 1, 1)
    
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    lda = go.Scatter(x=X_r2[y == i, 0], 
                     y=X_r2[y == i, 1],
                     showlegend=False,
                     mode='markers',
                     marker=dict(color=color),
                     name=target_name
                     )
    
    fig.append_trace(lda, 1, 2)
    
for i in map(str, range(1, 3)):
    x = 'xaxis' + i
    y = 'yaxis' + i
    
    fig['layout'][x].update(zeroline=False, showgrid=False)
    fig['layout'][y].update(zeroline=False, showgrid=False)
    
py.plot(fig)