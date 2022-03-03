## Libraries to use
import pandas as pd
import numpy as np
import pdb
import sklearn
from sklearn.decomposition import PCA
import statistics as statistics_formulas
import plotly.offline as opy
from matplotlib import pyplot as plt

## Load excel file with Peak Matrix
file_name = 'peakmatrix_lipids4.csv'
peak_matrix = pd.read_csv(file_name)

## Prepare data
data_intensities = peak_matrix.copy()
data_mass = data_intensities['Mass'].copy().tolist()
del data_intensities['Mass']
  
## Data scaled to unit variance and centered (mean = 0 and std = 1).
print("Scaling data...")
data_intensities.index = data_mass
data_intensities = data_intensities.T

data_intensities_scaled = pd.DataFrame(index = data_intensities.index, columns = data_intensities.columns).fillna(0)
for ncol in data_intensities.columns: 
    data_intensities_scaled[ncol] = (data_intensities[ncol] - np.mean(data_intensities[ncol])) / statistics_formulas.stdev(data_intensities[ncol])

data_intensities_scaled = data_intensities_scaled.T.fillna(0)

## PCA function    
print("Performing PCA... ")

n_samples = data_intensities_scaled.shape[1]
pca = PCA(n_components = n_samples-1)
coord_data_individuals = pca.fit_transform(data_intensities_scaled.T)
coord_data_individuals_df = pd.DataFrame(coord_data_individuals)

## Draw PC0 vs PC1
pc0_coords = coord_data_individuals_df.ix[:,0]
pc1_coords = coord_data_individuals_df.ix[:,1]
color_samples = ['#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#0000FF', '#FFA500', '#FFA500','#FFA500','#FFA500','#FFA500','#FFA500','#FFA500','#FFA500','#FA8072','#FA8072','#FA8072','#FA8072','#FA8072','#FA8072','#DAF7A6','#DAF7A6','#DAF7A6','#DAF7A6','#DAF7A6','#DAF7A6','#DAF7A6','#DAF7A6'] 

pc0_pc1 = []

for ix_p in range(n_samples):
    pc0_pc1.append({
        'x': [pc0_coords[ix_p]],
        'y': [pc1_coords[ix_p]],
        'name': data_intensities.index[ix_p],
        'type': 'scatter',
        'mode': 'markers',
        'marker': {'color': color_samples[ix_p], 'size': 10}
        })

layout_fmapInd = {
    'title': 'PCA - Factor Map (2D)',
    'xaxis': {'title':'PCo'},
    'xaxis': {'title':'PC1'},
}

opy.plot({'data': pc0_pc1, 'layout': layout_fmapInd})
n_clases=4


list1=[]
list2=[]
for i in range(1,n_clases+1):
    list1.append(i)
    list2.append('Type '+str(i))

label_dict = dict(zip(list1,list2))
y=[1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,4,4,4,4,4,4,4,4]
pc0_coords=np.asarray(pc0_coords)
pc1_coords=np.asarray(pc1_coords)
X=np.column_stack((pc0_coords,pc1_coords))


print (X)
print (X.shape)
def plot_scikit_lda(k, title):

    ax = plt.subplot(111)
    for label,marker in zip(
        range(1,n_clases+1),range(3,n_clases+4)):

        plt.scatter(x=k[:,0][y == label],
                    y=k[:,1][y == label] * -1, # flip the figure
                    marker=marker,
                    
                    alpha=0.5,
                    label=label_dict[label])
    
    
    plt.xlabel('PC1')
    plt.ylabel('PC0')

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

plot_scikit_lda(X, title='PCA')