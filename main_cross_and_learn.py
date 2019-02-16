from experiment import Pretraining_Cross_and_Learn

# Main function to train our novel model
# The name sets up a destiantion folder in path_results, which can be set in the config.yml file
# The parameter num_frames_flow sets the number of flow frames used in a pair,
# using num_frames_flow=10 yields the best results but might be slow due to hard drive read bottleneck.
# The layer parameter selects the features at which our cross-modal and diversity losses are applied,
# available choices are: 'conv5', 'pool5', 'fc6', 'fc7'.
# With similarity_scheme we can choose wether to use 'cosine' or 'euclidean' distance.

e = Pretraining_Cross_and_Learn(name='cross_and_learn', batch_size=30, epochs=200, learning_rate=0.01,
	arch='caffe_bn', layer='fc6', similarity_scheme='cosine',
	split_channels=True, time_flip=True, num_frames_flow=10)
e.run()