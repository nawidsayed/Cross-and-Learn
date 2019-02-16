from experiment import Pretraining_Concat, Pretraining_Cross_and_Learn
from experiment import Finetuning_AR_RGB, Finetuning_AR_OF

# e = Pretraining_Cross_and_Learn('cal_new', batch_size=30, epochs=200, learning_rate=0.01,
# 	num_frames_flow=1)
# e.run()

# e = Finetuning_AR_RGB('cal_new', name_finetuning ='ar_ucf_200_ft', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=0, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=25, batch_size=128, reset_fc6=True, split=1, reset_fc7=True)
# e.run()

e = Pretraining_Concat('concat_new', batch_size=30, epochs=200,
	data_key='ucf', norm='caffe_bn', learning_rate=0.01, high_motion=1, split_channels=True,
	time_flip=True, num_frames_flow=1, lr_decay_scheme=1, modalities=['rgb', 'of'])
e.run()

# Check num_frames and remove_mot
# e = Pretraining_Cross_and_Learn('caffe_bn_ours_ucf_cod_of', batch_size=30, epochs=200, source='l', 
# 	data_key='ucf', norm='caffe_bn_g2', max_shift=0, remove_motion=False, layer='fc6',
# 	num_frames=10, num_frames_cod=4, modalities=['cod', 'of'], union=False, high_motion=1, 
# 	split_channels=True, time_flip=True, similarity_scheme='cosine',
# 	negatives_same_domain=True, no_positive=False, lamb_norm=0, lr_decay_scheme=1, learning_rate=0.01,
# 	weight_pos=0.5, eps=0.0001, ada_weight_pos=False, ada_weight_pos_intervall=2, split=1, 
# 	dropout=0.5, gradient_dot='balanced', leaky_relu=False)
# e.run()


#####################################################
# CaffeNet finetuning
#####################################################

# e = Finetuning_AR_RGB('new_names', name_finetuning ='ar_hmdb_200_ft', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=25, batch_size=128, reset_fc6=True, split=1, reset_fc7=True)
# e.run()

# e = Finetuning_AR_RGB('new_names', name_finetuning ='ar_ucf_200_ft', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=0, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=25, batch_size=128, reset_fc6=True, split=1, reset_fc7=True)
# e.run()

# e = Finetuning_AR_OF('new_names', name_finetuning ='ar_ucf_200_flow', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True,
# 	reset_fc7=True, remove_motion=False)
# e.run()

# e = Finetuning_AR_OF('new_names', name_finetuning ='ar_hmdb_200_flow', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True,
# 	reset_fc7=True, remove_motion=False)
# e.run()

# e = Finetuning_ar_COD('test_remove', name_finetuning ='ar_ucf_200_cod', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True, reset_fc7=True)
# e.run()

# e = Finetuning_ar_COD('test_remove', name_finetuning ='ar_hmdb_200_cod', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True, reset_fc7=True)
# e.run()


#####################################################
# VGG16 finetuning
#####################################################

# e = Finetuning_AR_RGB('vgg16bn_ours_ucf', name_finetuning ='ar_ucf_200_2', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.005, num_test=5, batch_size=40, reset_fc7=True, reset_fc6=True)
# e.run(split_batch_test=5)

# e = Finetuning_AR_RGB('vgg16bn_ours_ucf', name_finetuning ='ar_hmdb_200_2', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.005, num_test=5, batch_size=40, reset_fc7=True, reset_fc6=True)
# e.run(split_batch_test=5)

# e = Finetuning_AR_OF('vgg16bn_ours_all', name_finetuning ='ar_ucf_400_flow', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, source='l', epochs=400,
# 	learning_rate=0.005, num_test=5, batch_size=40)
# e.run(split_batch_test=5)

# e = Finetuning_AR_OF('vgg16bn_ours_all', name_finetuning ='ar_hmdb_400_flow', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, source='l', epochs=400, 
# 	learning_rate=0.005, num_test=5, batch_size=40)
# e.run(split_batch_test=5)






