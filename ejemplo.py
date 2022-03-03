
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(4711)  # for repeatability of this tutorial

# Componentes x e y a clasterizar
X=[[1,3],[2,3],[9,9],[10,11]]
X=np.asarray(X)
print (X.shape)


Z = linkage(X, 'single')

fig = plt.figure(figsize=(25, 10))
dn = dendrogram(Z)
print (Z)


plt.show()

