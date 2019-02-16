from experiment import Finetuning_AR_RGB

# Main function to fine-tune the RGB network a pre-trained model with the given name
# The parameter name_finetuning sets up a new folder in the respective pre-training directory.
# Parameter split selects available train/test splits and can be 1, 2 or 3.
# load_epoch_pt selects the pre-training checkpoint epoch from which to load the model parameters,
# setting this to -1 loads the most recent checkpoint.

e = Finetuning_AR_RGB(name='cross_and_learn', name_finetuning ='finetuning_UCF_RGB', 
	split=1, epochs=200, batch_size=128, learning_rate=0.01, 
	load_epoch_pt=0)
e.run()


