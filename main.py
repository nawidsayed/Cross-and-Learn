from experiment import Pretraining_Concat, Pretraining_Cross_and_Learn
from experiment import Finetuning_AR_RGB, Finetuning_AR_OF

# ========================================================================================
# Run by uncomenting the respective experiment, multiple experimens can be run sucessively 
# ========================================================================================

# Classes to train our novel model or a Concat model
# The name sets up a destiantion folder in path_results, which can be set in the config.yml file
# The parameter num_frames_flow sets the number of flow frames used in a pair,
# using num_frames_flow=10 yields the best results but might be slow due to hard drive read bottleneck.
# The layer parameter selects the features at which our cross-modal and diversity losses are applied,
# available choices are: 'conv5', 'pool5', 'fc6', 'fc7'.
# With similarity_scheme we can choose wether to use 'cosine' or 'euclidean' distance.

# Pretraining_Cross_and_Learn(name='cross_and_learn', batch_size=30, epochs=200, learning_rate=0.01,
# 	arch='caffe_bn', layer='fc6', similarity_scheme='cosine',
# 	split_channels=True, time_flip=True, num_frames_flow=10).run()

# Pretraining_Concat(name='concat', batch_size=30, epochs=200, learning_rate=0.01,
# 	arch='caffe_bn', 
# 	split_channels=True, time_flip=True, num_frames_flow=10).run()

# Classes to fine-tune the RGB or optical flow network a pre-trained model with the given name
# The parameter name_finetuning sets up a new folder in the respective pre-training directory.
# Parameter split selects available train/test splits and can be 1, 2 or 3.
# load_epoch_pt selects the pre-training checkpoint epoch from which to load the model parameters,
# setting this to -1 loads the most recent checkpoint.

# Finetuning_AR_RGB(name='cross_and_learn', name_finetuning ='finetuning_UCF_RGB', 
# 	split=1, epochs=200, batch_size=128, learning_rate=0.01, 
# 	load_epoch_pt=-1).run()

Finetuning_AR_OF(name='cross_and_learn', name_finetuning ='finetuning_UCF_OF', 
	split=1, epochs=200, batch_size=128, learning_rate=0.01,
	load_epoch_pt=-1).run()

