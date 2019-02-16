from experiment import Pretraining_Concat

# Main function to train a Concat model
# The name sets up a destiantion folder in path_results, which can be set in the config.yml file
# The parameter num_frames_flow sets the number of flow frames used in a pair,
# using num_frames_flow=10 yields the best results but might be slow due to hard drive read bottleneck.

e = Pretraining_Concat(name='concat', batch_size=30, epochs=200, learning_rate=0.01,
	arch='caffe_bn', 
	split_channels=True, time_flip=True, num_frames_flow=10)
e.run()

