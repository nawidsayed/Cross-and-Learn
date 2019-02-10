id = 46
print('saved correctly %d' %id)

from experiment import Experiment_pretraining_def, Experiment_pretraining_fm
from experiment import Experiment_finetuning_ar_RGB, Experiment_finetuning_ar_OF

e = Experiment_pretraining_fm('cal_fm', batch_size=30, epochs=200, 
	data_key='ucf', norm='caffe_bn', layer='fc6',
	num_frames=1, num_frames_cod=4, modalities=['rgb', 'of'], high_motion=1, 
	split_channels=True, time_flip=True, similarity_scheme='cosine',
	learning_rate = 0.01, lr_decay_scheme=1, eps=0.0001, 
	dropout=0.5)
e.run()

e = Experiment_finetuning_ar_RGB('cal_fm', name_finetuning ='ar_ucf_200_ft', 
	dropout=0.5, data_key='ucf', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
	learning_rate=0.01, num_test=5, batch_size=128, reset_fc6=True, split=1, reset_fc7=True)
e.run()
e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_pretraining_def('new_names_def', batch_size=5, epochs=200,
# 	data_key='ucf', norm='caffe_bn', learning_rate=0.01, high_motion=1, split_channels=True,
# 	time_flip=True, num_frames=10, lr_decay_scheme=1, modalities=['rgb', 'of'])
# e.run()

# Check num_frames and remove_mot
# e = Experiment_pretraining_fm('caffe_bn_ours_ucf_cod_of', batch_size=30, epochs=200, source='l', 
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

# e = Experiment_finetuning_ar_RGB('new_names', name_finetuning ='ar_hmdb_200_ft', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=5, batch_size=128, reset_fc6=True, split=1, reset_fc7=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_RGB('new_names', name_finetuning ='ar_ucf_200_ft', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=0, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=5, batch_size=128, reset_fc6=True, split=1, reset_fc7=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_OF('new_names', name_finetuning ='ar_ucf_200_flow', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True,
# 	reset_fc7=True, remove_motion=False)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_OF('new_names', name_finetuning ='ar_hmdb_200_flow', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True,
# 	reset_fc7=True, remove_motion=False)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)


# e = Experiment_finetuning_ar_COD('test_remove', name_finetuning ='ar_ucf_200_cod', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True, reset_fc7=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_COD('test_remove', name_finetuning ='ar_hmdb_200_cod', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, source='l', epochs=200, reset_fc6=True, reset_fc7=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)




#####################################################
# VGG16 finetuning
#####################################################

# e = Experiment_finetuning_ar_RGB('vgg16bn_ours_ucf', name_finetuning ='ar_ucf_200_2', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.005, num_test=3, batch_size=40, reset_fc7=True, reset_fc6=True)
# e.run()
# e.evaluate_net(num_test=5, final_test_runs=2, split_batch=5)

# e = Experiment_finetuning_ar_RGB('vgg16bn_ours_ucf', name_finetuning ='ar_hmdb_200_2', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.005, num_test=3, batch_size=40, reset_fc7=True, reset_fc6=True)
# e.run()
# e.evaluate_net(num_test=5, final_test_runs=2, split_batch=5)


# e = Experiment_finetuning_ar_OF('vgg16bn_ours_all', name_finetuning ='ar_ucf_400_flow', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, source='l', epochs=400,
# 	learning_rate=0.005, num_test=3, batch_size=40)
# e.run()
# e.evaluate_net(num_test=5, final_test_runs=2, split_batch=5, load_epoch=400)

# e = Experiment_finetuning_ar_OF('vgg16bn_ours_all', name_finetuning ='ar_hmdb_400_flow', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, source='l', epochs=400, 
# 	learning_rate=0.005, num_test=3, batch_size=40)
# e.run()
# e.evaluate_net(num_test=5, final_test_runs=2, split_batch=5, load_epoch=400)

#####################################################
# NOt quite clear whats going on here 
#####################################################

# e = Experiment_finetuning_ar_RGB('vgg16bn_ours_ucf', 
# 	name_finetuning ='ar_ucf_200_transfer_rgb_student_drop_euc', name_experiment='transfer_rgb_student_drop_euc', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, epochs=200, freeze_layer='input', learning_rate=0.01,
# 	reset_fc7=True, reset_fc6=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_RGB('vgg16bn_ours_ucf', 
# 	name_finetuning ='ar_hmdb_200_transfer_rgb_student_drop_euc', name_experiment='transfer_rgb_student_drop_euc', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', learning_rate=0.01,
# 	reset_fc7=True, reset_fc6=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_RGB('vgg16bn_ours_ucf', 
# 	name_finetuning ='ar_ucf_200_transfer_rgb_student_drop_euc_2', name_experiment='transfer_rgb_student_drop_euc', 
# 	dropout=0.5, data_key='ucf', load_epoch_pt=-1, epochs=200, freeze_layer='input', learning_rate=0.01,
# 	reset_fc7=True, reset_fc6=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_RGB('vgg16bn_ours_ucf', 
# 	name_finetuning ='ar_hmdb_200_transfer_rgb_student_drop_euc_2', name_experiment='transfer_rgb_student_drop_euc', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', learning_rate=0.01,
# 	reset_fc7=True, reset_fc6=True)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)












# e = Experiment_finetuning_ar_RGB('caffe_bn_ours_ucf_ul30', name_finetuning ='ar_hmdb_200_3', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=5, batch_size=128, reset_fc6=True, split=1)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_RGB('caffe_bn_ours_ucf_ul30', name_finetuning ='ar_hmdb_200_s2', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=5, batch_size=128, reset_fc6=True, split=2)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)

# e = Experiment_finetuning_ar_RGB('caffe_bn_ours_ucf_ul30', name_finetuning ='ar_hmdb_200_s3', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=5, batch_size=128, reset_fc6=True, split=3)
# e.run()
# e.evaluate_net(num_test=25, final_test_runs=2)


# e = Experiment_finetuning_ar_RGB('alex_imagenet', name_finetuning ='ar_hmdb_200_resfc7', 
# 	dropout=0.5, data_key='hmdb', load_epoch_pt=-1, epochs=200, freeze_layer='input', 
# 	learning_rate=0.01, num_test=5, batch_size=128, reset_fc7=True)
# e.run()
# e.evaluate_net(num_test=5, final_test_runs=2)

# e = Experiment_finetuning_twostream('two_stream_test', 'alex_imagenet', 'caffe_bn_ours_ucf_ul30', 
# 	name_finetuning='two_stream_rgb', name_finetuning_1='ar_ucf_200_resfc7', name_finetuning_2='ar_ucf_200',
# 	data_key='ucf', fusion_scheme='avg')
# e.evaluate_net(num_test=5, final_test_runs=2)

# e = Experiment_finetuning_twostream('two_stream_test', 'alex_imagenet', 'caffe_bn_ours_ucf_ul30', 
# 	name_finetuning='two_stream_rgb', name_finetuning_1='ar_hmdb_200', name_finetuning_2='ar_hmdb_200',
# 	data_key='hmdb', fusion_scheme='avg')
# e.evaluate_net(num_test=5, final_test_runs=2)

# e = Experiment_finetuning_twostream('two_stream_alex', 'fm_l_caffe_bn_hm1_nsd', 'alex_imagenet', 
# 	name_finetuning='two_stream_rgb', name_finetuning_1='ar_ucf_200', name_finetuning_2='ar_ucf_200_resfc7',
# 	data_key='ucf', fusion_scheme='avg')
# e.evaluate_net(num_test=5, final_test_runs=2)





