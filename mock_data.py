import random

import numpy as np
import pandas as pd

import langbrainscore as lbs

import pdb

# ### Language brain-score example API usage notebook
# 
# we will use this notebook as an end-user example of how LBS may be used, assuming most of the LBS functionality is already implemented (whereas it may not be).
# In the meantime, the LBS interface will implement placeholder mock methods that either return `NotImplemented`, or otherwise return mock values consistent with types and dimensionality of what we would expect.
# Step 1 towards this direction is to create a mock dataset matching a real-world dataset in size and dimensionality.
# ## Generate mock dataset

# The Pereira (2018) [[pdf]](https://www.nature.com/articles/s41467-018-03068-4.pdf) [[data]](https://evlab.mit.edu/sites/default/files/documents/index2.html) [[supp OSF]](https://osf.io/crwz7/) dataset consists of `627 = (384 + 243)` stimuli

# define the size and dimensionality of mock data to create
num_stimuli = 627
num_neuroid = 10_000
# now randomly generate the mock data
recorded_data = np.random.rand(num_stimuli, num_neuroid)
print(recorded_data[:5,:5])
recorded_data.shape

recording_metadata = pd.DataFrame(dict(neuroid_id=[i for i in range(num_neuroid)],
                                       subj_id=[i % 100 for i in range(num_neuroid)]))


stimuli = [''.join(random.sample('abcdefghijklmnopqrstuvwxiz'*20, 7)) for _ in recorded_data]
print(len(stimuli), stimuli[:4], '...')


# instantiate a mock dataset 
fake_neuro_dataset = lbs.dataset.BrainDataset(stimuli, recorded_data, recording_metadata=recording_metadata)

mock_brain_encoder = lbs.interface.encoder.BrainEncoder()

brain_encoded_data = mock_brain_encoder.encode(fake_neuro_dataset)
print(brain_encoded_data, brain_encoded_data.shape)

# pdb.set_trace()

print('fitting a mapping')

ridge_mapping = lbs.mapping.Mapping('ridge')

output = ridge_mapping.map_cv(brain_encoded_data + np.random.rand(*brain_encoded_data.shape), 
                              brain_encoded_data[:,:400])

print(output[0], output[0][0].shape, output[0][1].shape)

metric = lbs.metrics.pearson_r.pearson_r

all_Y_pred = output[0][0]
all_Y_test = output[0][1]

# print(all_Y_pred)

pearson_rs = [metric(all_Y_pred[:, i], all_Y_test[:, i]) for i in range(all_Y_pred.shape[1])]
print(pearson_rs, len(pearson_rs))