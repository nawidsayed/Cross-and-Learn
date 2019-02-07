# Taken over and changed from torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import _pickle as pickle
from torchvision.models import vgg16
from torch.autograd import Variable


__all__ = ['Net_ar', 'VGG_16_BN',
    'get_network', 'Siamese', 'Siamese_fm', 'CaffeNet_BN']
    # 'get_arch',]

class Sim_func(object):
    def __init__(self, func, eps):
        self.func = func
        self.eps = eps

    def __call__(self, f_1, f_2):
        return self.func(f_1, f_2, eps=self.eps)

def cos_sim(f_1, f_2, eps=0.00001):
    len_1 = torch.sqrt(torch.sum(f_1 ** 2, dim=1) + eps)
    len_2 = torch.sqrt(torch.sum(f_2 ** 2, dim=1) + eps)
    return torch.sum(f_1 * f_2, dim=1) / (len_1 * len_2)

def euc_sim(f_1, f_2, eps=0.00001):
    sim = cos_sim(f_1, f_2, eps=eps)
    if not isinstance(sim, float):
        return 1 - torch.sqrt(2-2*sim)
    else:
        return 1 - np.sqrt(2-2*sim)

def normalize(arr):
    mini = np.min(arr)
    arr -= mini
    maxi = np.max(arr)
    arr /= maxi
    return arr, mini, maxi

def initialize_weights(net):
    return None
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, np.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, np.sqrt(2. / m.out_features))
            m.bias.data.zero_()

def set_dropout(net, dropout):
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.p = dropout

def get_network_dict(dict_info):
    Network = dict_info.pop('type')
    return Network(**dict_info)    

def get_network(path_dict):
    dict_info = pickle.load(open(path_dict, 'rb'))
    return get_network_dict(dict_info)

def get_arch(norm):
    if norm == 'alex':
        arch = AlexNet
    elif norm == 'caffe':
        arch = CaffeNet
    elif norm == 'caffe_bn':
        arch = CaffeNet_BN
    elif norm == 'caffe_bn_g2':
        arch =CaffeNet_BN
    elif norm == 'vgg':
        arch = VGG_Imagenet
    elif norm == 'vgg16':
        arch = VGG_16
    elif norm == 'vgg16bn':
        arch = VGG_16_BN
    elif norm == 'caffe_vgg_bn':
        arch = CaffeNet_BN
    else:
        arch = VGG
    return arch

def ml_drop(modulelist, drop):
    if drop > 0:
        length = drop
    else:
        length = len(modulelist)+drop
    ml_new = nn.ModuleList()
    for i in range(length):
        ml_new.append(modulelist[i])
    return ml_new

class Base_Network(nn.Module):
    def __init__(self):
        super(Base_Network, self).__init__()

    def load_state_dict_unconstrained(self, new_sd):
        sd = self.state_dict()
        unused_keys = []
        for key in new_sd:
            if key in sd:
                sd[key] = new_sd[key]
            else:
                unused_keys.append(key)
        if len(unused_keys) != 0:
            print('unused_keys: ', unused_keys)
        self.load_state_dict(sd)

    def get_net_info(self):
        return {'type': type(self)}

    def prep_tensor(self, x):
        return x

    def get_feature_output(self, x, layer='fc8'):
        x = self.prep_tensor(x)
        sub = 0
        if '_pre' in layer:
            layer = layer[:-4]
            sub = 1
        if not 'conv' in layer and not 'pool' in layer:
            x = self.get_features()(x)
            x = x.view(x.size(0), -1)
            x = self.get_classifier(drop=self.layer_dict[layer]-sub)(x)
        else:
            x = self.get_features(drop=self.layer_dict[layer]-sub)(x)
        return x

    def freeze_layers(self, layer='input'):
        if layer == 'input':
            pass
        elif not 'conv' in layer and not 'pool' in layer:
            for param in self.get_features().parameters():
                param.requires_grad = False
            for param in self.get_classifier(drop=self.layer_dict[layer]).parameters():
                param.requires_grad = False
        else:
            for param in self.get_features(drop=self.layer_dict[layer]).parameters():
                param.requires_grad = False

    def get_filters(self, arr=None):
        if arr is None:
            arr = self.get_features().state_dict()['0.weight'].cpu().numpy()
        arr, mini, maxi = normalize(arr)
        max_value = np.max([np.abs(mini), np.abs(maxi)])
        arr = np.swapaxes(arr,1,2)
        arr = np.swapaxes(arr,2,3)
        input_dim = self.input_dim
        ks = arr.shape[1]
        ks1 = ks+1
        wi, hi = self.tile_filters
        # if input_dim == 3:
        #     final_img = np.ones((hi*ks1, wi*ks1, 3))
        #     for i in range(hi):
        #         for j in range(wi):
        #             final_img[ks1*i:ks1*i+ks, ks1*j:ks1*j+ks, :] = arr[i*wi + j]        
        if input_dim % 3 == 0:
            nf = int(input_dim / 3)
            final_img = np.ones((nf*ks1, wi*hi*ks1, 3))
            for i in range(wi*hi):
                for j in range(nf):
                    final_img[ks1*j:ks1*j+ks, ks1*i:ks1*i+ks, :] = arr[i,:,:,3*j:3*(j+1)]
        else:
            final_img = np.ones((input_dim*ks1,wi*hi*ks1))
            nf = int(input_dim/2)
            for i in range(wi*hi):
                for j in range(nf):
                    final_img[ks1*j:ks1*j+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j]
                    final_img[ks1*(j+nf):ks1*(j+nf)+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j+1]
        return {'filter':final_img}

    def forward(self, x):
        return self.get_feature_output(x)

    def cod_to_coi(self):
        w_cod = self.get_features(drop=1)[0].weight.data
        w_coi = torch.Tensor(w_cod.size(0), w_cod.size(1)+3, w_cod.size(2), w_cod.size(3)).zero_()
        w_coi[:, 3:, :, :] += w_cod
        w_coi[:, :-3, :, :] -= w_cod
        w_cod = self.get_features(drop=1)[0].weight.data = w_coi
        self.input_dim += 3

    def get_features(self, drop=0):
        raise NotImplementedError('get_features not implemented in Base_Network')

    def get_classifier(self, drop=0):
        raise NotImplementedError('get_classifier not implemented in Base_Network')

    @property
    def input_spatial_size(self):
        raise NotImplementedError('Base_Network should implement input_spatial_size (tuple)')

    @property
    def input_dim(self):
        raise NotImplementedError('Base_Network should implement input_dim (int)')

    @property
    def layer_dict(self):
        raise NotImplementedError('Base_Network should implement layer_dict (dict)')

    @property
    def tile_filters(self):
        raise NotImplementedError('Base_Network should implement tile_filters ((int, int))')



class CaffeNet_BN(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    layer_dict = {'pool2':8,'conv3':10, 'conv4':13, 'conv5':16, 'pool5':18, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True, leaky_relu=False, groups=2):
        super(CaffeNet_BN, self).__init__()
        self.input_dim = input_dim
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 6x6
        ])
        nonlinearity = nn.ReLU(inplace=True)
        if leaky_relu:
            nonlinearity = nn.LeakyReLU(inplace=True)
        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nonlinearity,
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000)
        ])
        if init:
            initialize_weights(self)

    def prep_tensor(self, x):
        return x

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(CaffeNet_BN, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim})
        return dict_info

    def reset_fc7(self):
        m = self.classifier[4]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    def reset_fc6(self):
        m = self.classifier[1]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class VGG_16_BN(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    layer_dict = {'pool2':34,'conv3':37, 'conv4':40, 'conv5':43, 'pool5':44, 'fc6':3, 'fc7':6}
    tile_filters = (8,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True, groups=1, leaky_relu=False):
        super(VGG_16_BN, self).__init__()
        self.input_dim = input_dim
        self.features = nn.ModuleList([
            # 224x224
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1),    # 0                               
            nn.BatchNorm2d(64),                                                                    
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 64, kernel_size=3, padding=1),                # 3           
            nn.BatchNorm2d(64),                        
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2, stride=2),                         
            # 112x112                          
            nn.Conv2d(64, 128, kernel_size=3, padding=1),               # 7        
            nn.BatchNorm2d(128),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(128, 128, kernel_size=3, padding=1),              # 10           
            nn.BatchNorm2d(128),                           
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2, stride=2),                         
            # 56x56                        
            nn.Conv2d(128, 256, kernel_size=3, padding=1),              # 14       
            nn.BatchNorm2d(256),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(256, 256, kernel_size=3, padding=1),              # 17       
            nn.BatchNorm2d(256),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(256, 256, kernel_size=3, padding=1),              # 20                        
            nn.BatchNorm2d(256),                           
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2, stride=2),                      
            # 28x28                        
            nn.Conv2d(256, 512, kernel_size=3, padding=1),              # 24           
            nn.BatchNorm2d(512),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # 27                        
            nn.BatchNorm2d(512),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # 30       
            nn.BatchNorm2d(512),                           
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2, stride=2),                         
            # 14x14                        
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # 34       
            nn.BatchNorm2d(512),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # 37       
            nn.BatchNorm2d(512),                           
            nn.ReLU(inplace=True),                         
            nn.Conv2d(512, 512, kernel_size=3, padding=1),              # 40                           
            nn.BatchNorm2d(512),                           
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2, stride=2)                          
            # 7x7
        ])
        nonlinearity = nn.ReLU(inplace=True)
        if leaky_relu:
            nonlinearity = nn.LeakyReLU(inplace=True)
        self.classifier = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nonlinearity,
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 2)
        ])

        if init:
            initialize_weights(self)

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(VGG_16_BN, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim})
        return dict_info

    def reset_fc7(self):
        m = self.classifier[3]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    def reset_fc6(self):
        m = self.classifier[0]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

# feature_net is of type Base_Network with get_fc7_features returning 4096dim vector 
class Net_ar(Base_Network):
    input_spatial_size = None
    input_dim = None
    data_keys_dict = {'ucf':101, 'hmdb':51, 'act':43, 'olympic':16, 'all':195}
    def __init__(self, feature_net, dropout=0.5, data_key='ucf'):
        super(Base_Network, self).__init__()
        if isinstance(feature_net, Base_Network):
            self.feature_net = feature_net
        else:
            self.feature_net = get_network_dict(feature_net)
        self.input_spatial_size = self.feature_net.input_spatial_size
        self.input_dim = self.feature_net.input_dim
        self.data_key = data_key
        if data_key in self.data_keys_dict:
            output_dim = self.data_keys_dict[data_key]
        else:
            output_dim = data_key

        self.classifier_ar = nn.Linear(4096, output_dim)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout

    def forward(self, x):
        x = self.feature_net.get_feature_output(x, layer='fc7')
        return self.classifier_ar(x)

    def get_features(self, drop=0):
        return self.feature_net.get_features(drop=drop)

    def get_classifier(self, drop=0):
        if drop == 0 or drop == 7:
            return nn.Sequential(self.feature_net.get_classifier(-1), self.classifier_ar)
        else:
            return nn.Sequential(self.feature_net.get_classifier(drop))

    def get_net_info(self):
        dict_info = super(Net_ar, self).get_net_info()
        dict_info.update({'feature_net': self.feature_net.get_net_info(), 'data_key': self.data_key})
        return dict_info

    def get_filters(self):
        return self.feature_net.get_filters()

    def get_feature_output(self, x, layer='fc8'):
        if layer == 'fc8':
            x = self.feature_net.get_feature_output(x, layer='fc7')
            return self.classifier_ar(x) 
        else:
            return self.feature_net.get_feature_output(x, layer)

    def cod_to_coi(self):
        super(Net_ar, self).cod_to_coi()
        self.feature_net.input_dim += 3

    def freeze_layers(self, layer='input'):
        if layer == 'fc8':
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.feature_net.freeze_layers(layer)

    def reset_fc7(self):
        self.feature_net.reset_fc7()

    def reset_fc6(self):
        self.feature_net.reset_fc6()


class Base_Siamese(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    def __init__(self, norm='BN', num_frames=12, num_frames_cod=4, dropout=0.5, 
        modalities=['rgb', 'of'], decoder=0, leaky_relu=False):
        super(Base_Siamese, self).__init__()
        if norm != 'caffe_bn_g2' and norm != 'vgg16bn' and norm != 'caffe_bn':
            raise Exception('leaky relu not implemented for other network architectures')
        self.norm = norm
        self.num_frames = num_frames
        self.num_frames_cod = num_frames_cod
        self.modalities = modalities
        if self.norm == 'caffe_bn':
            arch = CaffeNet_BN
        elif self.norm == 'vgg16bn':
            arch = VGG_16_BN

        if 'rgb' in self.modalities:
            self.app_net = arch(input_dim=3, dropout=dropout, leaky_relu=leaky_relu)
        if 'of' in self.modalities:
            self.mot_net = arch(input_dim=num_frames*2, dropout=dropout, leaky_relu=leaky_relu)
        if 'cod' in self.modalities:
            self.cod_net = arch(input_dim=num_frames_cod*3, dropout=dropout,
                leaky_relu=leaky_relu)


    # def get_app_net(self):
    #     return self.app_net

    # def get_mot_net(self):
    #     return self.mot_net

    # def get_cod_net(self):
    #     return self.cod_net

    # def set_app_net(self, app_net):
    #     self.app_net = app_net

    def get_net_info(self):
        dict_info = super(Base_Siamese, self).get_net_info()
        dict_info.update({'norm': self.norm, 'num_frames': self.num_frames, 
            'num_frames_cod': self.num_frames_cod, 'modalities':self.modalities})
        return dict_info

    def get_filters(self):
        dict_filter = {}
        if 'rgb' in self.modalities:
            dict_filter['app'] = self.app_net.get_filters()['filter']
        if 'of' in self.modalities:
            dict_filter['mot'] = self.mot_net.get_filters()['filter']
        if 'cod' in self.modalities:
            dict_filter['cod'] = self.cod_net.get_filters()['filter']
        return dict_filter

    def get_features(self, drop=0):
        raise NotImplementedError('previous implementation deprecated')
        # return self.app_net.get_features(drop=drop), self.mot_net.get_features(drop=drop)

class Siamese(Base_Siamese):
    def __init__(self, norm='BN', num_frames=12, num_frames_cod=4, dropout=0.5, modalities=['rgb', 'of'],
        decoder=None, layer='fc6'):
        super(Siamese, self).__init__(norm=norm, num_frames=num_frames, dropout=dropout, 
            modalities=modalities, num_frames_cod=num_frames_cod)
        self.layer = layer
        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(2 * 4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        ])

    def forward(self, *sample):
        sample = list(sample)
        features = []
        nonzeros = []
        for i in range(2):
            if 'rgb' in self.modalities:
                image = sample.pop(0)
                image_feat = self.app_net.get_feature_output(image, layer=self.layer)
                image_feat = image_feat.view(image_feat.size(0), -1)
                features.append(image_feat)
                nonzeros.append(image_feat.data.nonzero().size(0))
            if 'of' in self.modalities:
                flow = sample.pop(0)
                flow_feat = self.mot_net.get_feature_output(flow, layer=self.layer)
                flow_feat = flow_feat.view(flow_feat.size(0), -1)
                features.append(flow_feat)
                nonzeros.append(flow_feat.data.nonzero().size(0))
            if 'cod' in self.modalities:
                cod = sample.pop(0)
                cod_feat = self.cod_net.get_feature_output(cod, layer=self.layer)
                cod_feat = cod_feat.view(cod_feat.size(0), -1)
                features.append(cod_feat)
                nonzeros.append(cod_feat.data.nonzero().size(0))

        outputs = []
        outputs.append(torch.cat([features[0], features[1]], dim=1))
        outputs.append(torch.cat([features[2], features[3]], dim=1))
        outputs.append(torch.cat([features[0], features[3]], dim=1))
        outputs.append(torch.cat([features[2], features[1]], dim=1))
        predictions = []
        for output in outputs:
            for module in self.classifier:
                output = module(output)
            predictions.append(output)
        return predictions[0], predictions[1], predictions[2], predictions[3], nonzeros       

class Siamese_fm(Base_Siamese):
    def __init__(self, norm='BN', layer='fc6', num_frames=12, num_frames_cod=4, dropout=0.5,
        modalities=['rgb', 'of'], union=False, decoder=False, similarity_scheme='cosine', 
        leaky_relu=False, eps=0.001, ):
        super(Siamese_fm, self).__init__(norm=norm, num_frames=num_frames, 
            num_frames_cod=num_frames_cod, dropout=dropout, modalities=modalities,
            decoder=decoder, leaky_relu=leaky_relu)
        if leaky_relu and layer != 'fc6':
            raise Exception('leaky relu currently only at fc6 implemented')
        self.layer = layer
        self.union = union
        self.similarity_scheme = similarity_scheme
        self.eps = eps
        self._set_similarity_func()

    def forward(self, *sample):
        sample = list(sample)
        features = []
        lengths = []
        nonzeros = []
        for i in range(2):
            if 'rgb' in self.modalities:
                image = sample.pop(0)
                image_feat = self.app_net.get_feature_output(image, layer=self.layer)
                image_feat = image_feat.view(image_feat.size(0), -1)
                features.append(image_feat)
                lengths.append((image_feat ** 2).sum(dim=1).sqrt())
                nonzeros.append(image_feat.data.nonzero().size(0))
            if 'of' in self.modalities:
                flow = sample.pop(0)
                flow_feat = self.mot_net.get_feature_output(flow, layer=self.layer)
                flow_feat = flow_feat.view(flow_feat.size(0), -1)
                features.append(flow_feat)
                lengths.append((flow_feat ** 2).sum(dim=1).sqrt())
                nonzeros.append(flow_feat.data.nonzero().size(0))
            if 'cod' in self.modalities:
                cod = sample.pop(0)
                cod_feat = self.cod_net.get_feature_output(cod, layer=self.layer)
                cod_feat = cod_feat.view(cod_feat.size(0), -1)
                features.append(cod_feat)
                lengths.append((cod_feat ** 2).sum(dim=1).sqrt())
                nonzeros.append(cod_feat.data.nonzero().size(0))


        sim_true_1 = self.sim_func(features[0], features[1]) 
        sim_true_2 = self.sim_func(features[2], features[3]) 
        sim_false_1 = self.sim_func(features[0], features[2]) 
        sim_false_2 = self.sim_func(features[1], features[3])                
        final_sim_true_1 = (sim_true_1 + sim_true_2) / 2
        final_sim_false_1 = (sim_false_1 + sim_false_2) / 2

        return sim_true_1, sim_true_2, sim_false_1, sim_false_2, nonzeros


    def set_layer(self, layer):
        self.layer = layer

    def get_net_info(self):
        dict_info = super(Siamese_fm, self).get_net_info()
        dict_info.update({'layer': self.layer})
        return dict_info

    def _set_similarity_func(self):
        if self.similarity_scheme == 'cosine':
            self.sim_func = Sim_func(cos_sim, self.eps)
        if self.similarity_scheme == 'euclidean':
            self.sim_func = Sim_func(euc_sim, self.eps)

    def get_feature_output(self, *sample, layer='fc6', dismissed=[]):
        sample = list(sample)
        features = []
        for i in range(2):
            if 'rgb' in self.modalities:
                image = sample.pop(0)
                if 'rgb' in dismissed:
                    image_feat = None
                else:
                    image_feat = self.app_net.get_feature_output(image, layer=layer)
                features.append(image_feat)
            if 'of' in self.modalities:
                flow = sample.pop(0)
                if 'of' in dismissed:
                    flow_feat = None
                else:
                    flow_feat = self.mot_net.get_feature_output(flow, layer=layer)
                features.append(flow_feat)
            if 'cod' in self.modalities:
                cod = sample.pop(0)
                if 'cod' in dismissed:
                    cod_feat = None
                else:
                    cod_feat = self.cod_net.get_feature_output(cod, layer=layer)
                features.append(cod_feat)
        return features








if __name__ == '__main__':
    net_1 = Single_def(norm='alex')
    print('net_1.app_net before: ', net_1.app_net.features[0].bias[:5])
    net_2 = Siamese_fm(norm='alex')
    print('net_2.app_net before: ', net_2.app_net.features[0].bias[:5])
    print('net_2.mot_net before: ', net_2.mot_net.features[0].bias[:5])
    net = Container(net_1, net_2)
    net = net.cuda()
    net_1.app_net.features[0].bias[:5] *= 2
    print('net_1.app_net after: ', net_1.app_net.features[0].bias[:5])
    print('net_2.app_net after: ', net_2.app_net.features[0].bias[:5])
    print('net_2.mot_net after: ', net_2.mot_net.features[0].bias[:5])


