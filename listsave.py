import pickle
import numpy as np #using pickle
features = []
labels = []

with open('features.pkl', 'wb') as f:
    pickle.dump(features, f)

#np.save('labels.npy', labels)
with open('labels.pkl', 'wb') as f:
    pickle.dump(labels, f)
