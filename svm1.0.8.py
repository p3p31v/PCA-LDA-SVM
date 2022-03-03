import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import statistics as statistics_formulas
import pylab as pl
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import feature_selection


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



Y=[1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4]

X_new = SelectKBest(chi2, k=2).fit_transform(X, Y)


Y=np.asarray(Y)

print (X_new)
print (Y.shape)
k = list(set(Y))
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
print(Y.shape)
X=X_new
h = .02  # step size in the mesh


C = 1.0  # SVM regularization parameter
print (X)

svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
rbf_svc = svm.SVC(kernel='rbf', C=C).fit(X, Y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)
print (X.shape)



# create a mesh to plot in


x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1



xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
          'LinearSVC (linear kernel)']

for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    pl.subplot(2, 2, i + 1)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.contourf(xx, yy, Z, cmap=pl.cm.coolwarm, alpha=0.8)
    pl.axis('off')

    # Plot also the training points
    pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.coolwarm)

    pl.title(titles[i])

pl.show()