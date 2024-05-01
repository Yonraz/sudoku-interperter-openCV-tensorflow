import numpy as np

# Now load the data
with np.load('knn_data.npz') as data:
 print( data.files )
 train = data['train']
 train_labels = data['train_labels']