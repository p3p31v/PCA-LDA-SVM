from sklearn.feature_selection import RFE
import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import statistics as statistics_formulas
import pylab as pl
from sklearn.feature_selection import RFE
from sklearn.svm import SVC

## applying rfe to linear model considering only the first two features
model = svm.SVC(kernel="linear")
rfe = RFE(model, 2)
## loading the data
file_name = 'EFSvsEFVanBvsEFVanA_FindPeaks40_mt5.csv'
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
Y=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]




rfe = rfe.fit(X, Y)

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


svc = svm.SVC(kernel='linear', C=C).fit(X, Y)
lin_svc = svm.LinearSVC(C=C).fit(X, Y)




# create a mesh to plot in


x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1



xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)']

for i, clf in enumerate((svc, lin_svc)):
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