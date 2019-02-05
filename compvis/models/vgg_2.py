# Taken over and changed from torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import _pickle as pickle
from torchvision.models import vgg16
from torch.autograd import Variable


__all__ = ['VGG', 'Net_ar', 'AlexNet', 'CaffeNet', 'VGG_Imagenet', 'VGG_16', 'VGG_16_BN', 'Two_Stream',
    'get_network', 'Siamese', 'Siamese_fm', 'Single_def', 'Single_fm', 'Container', 'CaffeNet_BN',
    'get_arch', 'CaffeNet_NLM', 'CaffeNet_3BN', 'Caffe_biagio']

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

def euc_nonorm_sim(f_1, f_2, eps=0.00001):
    dist = torch.sum(torch.sqrt((f_1-f_2) ** 2), dim=1)
    return -dist

def lin_sim(f_1, f_2, eps=0.00001):
    sim = cos_sim(f_1, f_2, eps=eps)
    if not isinstance(sim, float):
        return 1 - 2*torch.acos(sim) / np.pi
    else:
        return 1 - 2*np.arccos(sim) / np.pi

def own_sim(f_1, f_2, eps=0.00001):
    sim = cos_sim(f_1, f_2, eps=eps)
    return torch.cos(np.pi/2+torch.acos(sim))+1

def normalize(arr):
    mini = np.min(arr)
    arr -= mini
    maxi = np.max(arr)
    arr /= maxi
    return arr, mini, maxi

def downsampling(x, size=None, scale_factor=None, mode='nearest'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
    # create coordinates
    h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
    # create grid
    grid = torch.zeros(size[0],size[1],2)
    grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda: grid = grid.cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)

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

class LRN(nn.Module):
    def __init__(self, local_size=5, alpha=0.0001, beta=0.75, k=2):
        super(LRN, self).__init__()
        self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                stride=1,
                padding=(int((local_size-1.0)/2), 0, 0))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        div = x.mul(x).unsqueeze(1)
        div = self.average(div).squeeze(1)
        div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class NLM(nn.Module):
    def forward(self, x):
        return x

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


class Decoder_large(Base_Network):
    input_spatial_size = (1,1)
    output_spatial_size = (48, 48)
    input_dim = None
    def __init__(self, input_dim=4096, output_dim=3, dropout=0.5, init=True):
        super(Decoder_large, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(input_dim, 256 * 6 * 6),
            nn.ReLU(inplace=True),
        ])
        self.features = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # 12x12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # 24x24
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # 48x48
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
        ])
        if init:
            initialize_weights(self)

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for module in self.classifier:
            x = module(x)
        x = x.view(x.size(0), 256, 6, 6)
        for module in self.features:
            x = module(x)
        return x

    def get_net_info(self):
        dict_info = super(Decoder_large, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim, 'output_dim': self.output_dim})
        return dict_info

class Decoder_small(Base_Network):
    input_spatial_size = (1,1)
    output_spatial_size = (24, 24)
    input_dim = None
    def __init__(self, input_dim=4096, output_dim=3, dropout=0.5, init=True):
        super(Decoder_small, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(input_dim, 256 * 6 * 6),
            nn.ReLU(inplace=True),
        ])
        self.features = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # 12x12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            # 24x24
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
        ])
        if init:
            initialize_weights(self)

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for module in self.classifier:
            x = module(x)
        x = x.view(x.size(0), 256, 6, 6)
        for module in self.features:
            x = module(x)
        return x

    def get_net_info(self):
        dict_info = super(Decoder_small, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim, 'output_dim': self.output_dim})
        return dict_info



# images should have mean=0 and stdev=1 across all channels
# Using Alexnet-OWT
class AlexNet(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    layer_dict = {'pool2':6,'conv3':8, 'conv4':10, 'conv5':12, 'pool5':13, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (8,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True):
        super(AlexNet, self).__init__()
        self.input_dim = input_dim
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 6x6
        ])
        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000)
        ])
        if init:
            initialize_weights(self)

    def prep_tensor(self, x):
        return x
        # if x.size(1) < 3:
        #     return x
        # y = x.clone()
        # y[:,0] = (x[:,0] - self.mean[0]) / self.std[0]
        # y[:,1] = (x[:,1] - self.mean[1]) / self.std[1]
        # y[:,2] = (x[:,2] - self.mean[2]) / self.std[2]
        # return y

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(AlexNet, self).get_net_info()
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

    # def get_conv3_features(self, x):
    #     x = self.prep_tensor(x)
    #     for i in range(8):
    #         module = self.features[i]
    #         x = module(x)
    #     return x

    # def get_conv4_features(self, x):
    #     x = self.prep_tensor(x)
    #     for i in range(10):
    #         module = self.features[i]
    #         x = module(x)
    #     return x

    # def get_convpp5_features(self, x):
    #     x = self.prep_tensor(x)
    #     for i in range(12):
    #         module = self.features[i]
    #         x = module(x)
    #     return x

    # def get_conv5_features(self, x):
    #     x = self.prep_tensor(x)
    #     for module in self.features:
    #         x = module(x)
    #     return x

    # def get_fcpp6_features(self, x):
    #     x = self.get_conv5_features(x)
    #     x = x.view(x.size(0), -1)
    #     for i in range(2):
    #         module = self.classifier[i]
    #         x = module(x)
    #     return x

    # def get_fc6_features(self, x):
    #     x = self.get_conv5_features(x)
    #     x = x.view(x.size(0), -1)
    #     for i in range(3):
    #         module = self.classifier[i]
    #         x = module(x)
    #     return x

    # def get_fc7_features(self, x):
    #     x = self.get_conv5_features(x)
    #     x = x.view(x.size(0), -1)
    #     for i in range(6):
    #         module = self.classifier[i]
    #         x = module(x)
    #     return x

    # def forward(self, x):
    #     x = self.get_conv5_features(x)
    #     x = x.view(x.size(0), -1)
    #     for module in self.classifier[i]:
    #         x = module(x)
    #     return x

    # def get_filters(self):
    #     arr = self.get_features().state_dict()['0.weight'].cpu().numpy()
    #     arr, mini, maxi = normalize(arr)
    #     max_value = np.max([np.abs(mini), np.abs(maxi)])
    #     arr = np.swapaxes(arr,1,2)
    #     arr = np.swapaxes(arr,2,3)
    #     input_dim = self.input_dim
    #     ks = arr.shape[1]
    #     ks1 = ks+1
    #     wi, hi = (8, 8)
    #     if input_dim == 3:
    #         final_img = np.ones((hi*ks1, wi*ks1,3))
    #         for i in range(hi):
    #             for j in range(wi):
    #                 final_img[ks1*i:ks1*i+ks, ks1*j:ks1*j+ks, :] = arr[i*wi + j]          
    #     else:
    #         final_img = np.ones((input_dim*ks1,wi*hi*ks1))
    #         nf = int(input_dim/2)
    #         for i in range(wi*hi):
    #             for j in range(nf):
    #                 final_img[ks1*j:ks1*j+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j]
    #                 final_img[ks1*(j+nf):ks1*(j+nf)+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j+1]
    #     return {'filter':final_img}


class CaffeNet(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    layer_dict = {'pool2':8,'conv3':10, 'conv4':12, 'conv5':14, 'pool5':15, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True):
        super(CaffeNet, self).__init__()
        self.input_dim = input_dim
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 6x6
        ])
        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000)
        ])
        if init:
            initialize_weights(self)

    def prep_tensor(self, x):
        return x
        # if x.size(1) < 3:
        #     return x
        # y = x.clone()
        # y[:,0] = (x[:,0] - self.mean[0]) / self.std[0]
        # y[:,1] = (x[:,1] - self.mean[1]) / self.std[1]
        # y[:,2] = (x[:,2] - self.mean[2]) / self.std[2]
        # return y

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(CaffeNet, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim})
        return dict_info

    def reset_fc7(self):
        m = self.classifier[4]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class CaffeNet_3BN(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    layer_dict = {'pool2':6,'conv3':8, 'conv4':11, 'conv5':14, 'pool5':16, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True, groups=1):
        super(CaffeNet_3BN, self).__init__()
        self.input_dim = input_dim
        self.groups = groups
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=self.groups),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=self.groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=self.groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 6x6
        ])
        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000)
        ])
        if init:
            initialize_weights(self)

    def prep_tensor(self, x):
        return x
        # if x.size(1) < 3:
        #     return x
        # y = x.clone()
        # y[:,0] = (x[:,0] - self.mean[0]) / self.std[0]
        # y[:,1] = (x[:,1] - self.mean[1]) / self.std[1]
        # y[:,2] = (x[:,2] - self.mean[2]) / self.std[2]
        # return y

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(CaffeNet_3BN, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim, 'groups': self.groups})
        return dict_info

    def reset_fc7(self):
        m = self.classifier[4]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    def reset_fc6(self):
        m = self.classifier[1]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class CaffeNet_BN(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    layer_dict = {'pool2':8,'conv3':10, 'conv4':13, 'conv5':16, 'pool5':18, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True, groups=1, leaky_relu=False):
        super(CaffeNet_BN, self).__init__()
        self.input_dim = input_dim
        self.groups = groups
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(96),
            # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=self.groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=self.groups),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=self.groups),
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
        # if x.size(1) < 3:
        #     return x
        # y = x.clone()
        # y[:,0] = (x[:,0] - self.mean[0]) / self.std[0]
        # y[:,1] = (x[:,1] - self.mean[1]) / self.std[1]
        # y[:,2] = (x[:,2] - self.mean[2]) / self.std[2]
        # return y

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(CaffeNet_BN, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim, 'groups': self.groups})
        return dict_info

    def reset_fc7(self):
        m = self.classifier[4]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    def reset_fc6(self):
        m = self.classifier[1]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class CaffeNet_NLM(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    layer_dict = {'pool2':8,'conv3':10, 'conv4':13, 'conv5':16, 'pool5':18, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True, groups=1):
        super(CaffeNet_NLM, self).__init__()
        self.input_dim = input_dim
        self.groups = groups
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            NLM(),
            # 55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 27x27
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=self.groups),
            nn.ReLU(inplace=True),
            NLM(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 13x13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            NLM(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=self.groups),
            nn.ReLU(inplace=True),
            NLM(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=self.groups),
            nn.ReLU(inplace=True),
            NLM(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 6x6
        ])
        self.classifier = nn.ModuleList([
            nn.Dropout(dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000)
        ])
        if init:
            initialize_weights(self)

    def prep_tensor(self, x):
        return x
        # if x.size(1) < 3:
        #     return x
        # y = x.clone()
        # y[:,0] = (x[:,0] - self.mean[0]) / self.std[0]
        # y[:,1] = (x[:,1] - self.mean[1]) / self.std[1]
        # y[:,2] = (x[:,2] - self.mean[2]) / self.std[2]
        # return y

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(CaffeNet_NLM, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim, 'groups': self.groups})
        return dict_info

    def reset_fc7(self):
        m = self.classifier[4]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

    def reset_fc6(self):
        m = self.classifier[1]
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

# images should have mean=0 std=[60,60,70]
class VGG_Imagenet(Base_Network):
    # Reference to Cadene
    input_spatial_size = (224, 224)
    input_dim = None
    mean = np.array([123.68, 116.779, 103.939])
    layer_dict = {'pool2':8,'conv3':10, 'conv4':12, 'conv5':14, 'pool5':15, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, dropout=0.5, init=True):
        super(VGG_Imagenet, self).__init__()
        self.input_dim = input_dim
        self.features = nn.ModuleList([
            nn.Conv2d(self.input_dim,96,(7, 7),(2, 2)),
            nn.ReLU(),
            LRN(alpha=0.0005),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(96,256,(5, 5),(2, 2),(1, 1)),
            nn.ReLU(),
            LRN(alpha=0.0005),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        ])
        self.classifier = nn.ModuleList([
            nn.Linear(6*6*512,4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4096,1000)
        ])

    def prep_tensor(self, x):
        if x.size(1) < 3:
            return x
        y = x.clone()
        y[:,0] = x[:,0] * 255 - self.mean[0]
        y[:,1] = x[:,1] * 255 - self.mean[1]
        y[:,2] = x[:,2] * 255 - self.mean[2]
        return y

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(VGG_Imagenet, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim})
        return dict_info

    # def get_conv3_features(self, x):
    #     x = self.prep_tensor(x)
    #     for i in range(10):
    #         module = self.features[i]
    #         x = module(x)
    #     return x

    # def get_conv4_features(self, x):
    #     x = self.prep_tensor(x)
    #     for i in range(12):
    #         module = self.features[i]
    #         x = module(x)
    #     return x

    # def get_convpp5_features(self, x):
    #     x = self.prep_tensor(x)
    #     for i in range(14):
    #         module = self.features[i]
    #         x = module(x)
    #     return x

    # def get_conv5_features(self, x):
    #     x = self.prep_tensor(x)
    #     for module in self.features:
    #         x = module(x)
    #     return x

    # def get_fcpp6_features(self, x):
    #     x = self.get_conv5_features(x)
    #     x = x.view(x.size(0), -1)
    #     for i in range(2):
    #         module = self.classifier[i]
    #         x = module(x)
    #     return x

    # def get_fc6_features(self, x):
    #     x = self.get_conv5_features(x) 
    #     x = x.view(x.size(0), -1)
    #     for i in range(3):
    #         module = self.classifier[i]
    #         x = module(x)
    #     return x

    # def get_fc7_features(self, x):
    #     x = self.get_conv5_features(x) 
    #     x = x.view(x.size(0), -1)
    #     for i in range(6):
    #         module = self.classifier[i]
    #         x = module(x)
    #     return x

    # def forward(self, x):
    #     x = self.get_conv5_features(x) 
    #     x = x.view(x.size(0), -1)
    #     for module in self.classifier:
    #         x = module(x)
    #     return x  

    # def get_filters(self):
    #     arr = self.get_features().state_dict()['0.weight'].cpu().numpy()
    #     arr, mini, maxi = normalize(arr)
    #     max_value = np.max([np.abs(mini), np.abs(maxi)])
    #     arr = np.swapaxes(arr,1,2)
    #     arr = np.swapaxes(arr,2,3)
    #     input_dim = self.input_dim
    #     ks = arr.shape[1]
    #     ks1 = ks+1
    #     wi, hi = (12, 8)
    #     if input_dim == 3:
    #         final_img = np.ones((hi*ks1, wi*ks1,3))
    #         for i in range(hi):
    #             for j in range(wi):
    #                 final_img[ks1*i:ks1*i+ks, ks1*j:ks1*j+ks, :] = arr[i*wi + j]          
    #     else:
    #         final_img = np.ones((input_dim*ks1,wi*hi*ks1))
    #         nf = int(input_dim/2)
    #         for i in range(wi*hi):
    #             for j in range(nf):
    #                 final_img[ks1*j:ks1*j+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j]
    #                 final_img[ks1*(j+nf):ks1*(j+nf)+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j+1]
    #     return {'filter':final_img}




# images should have mean=[0.485, 0.456, 0.406] std=[0.229, 0.224, 0.225]
# class VGG_Conv(Base_Network):
#     input_spatial_size = (224, 224)
#     input_dim = None
#     layer_dict = {'pool2':8,'conv3':10, 'conv4':12, 'conv5':14, 'pool5':15, 'fc6':3,'fc7':6, 'fc8':7}
#     def __init__(self, input_dim=3, norm='BN', init=True):
#         super(VGG_Conv, self).__init__()
#         self.input_dim = input_dim
#         self.norm = norm
#         # if norm == 'LRN' or norm == 'BN':
#         if norm == 'BN':
#             norms = [nn.BatchNorm2d(96), nn.BatchNorm2d(256)]
#         elif norm == 'LRN':
#             norms = [LRN(), LRN()]
#         elif norm == 'NLM':
#             norms = [NLM(), NLM()]
#         self.features_1 = nn.ModuleList([
#     		# 224x224
#     		nn.Conv2d(input_dim, 96, kernel_size=7, stride=2, padding=0),
#     		nn.ReLU(True),
#             norms[0],
#     		# 109x109
#     		nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#     		# 55x55
#     		nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
#     		nn.ReLU(True),
#             norms[1],
#     		# 28x28
#     		nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#     		# 14x14
#     		nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#     		nn.ReLU(True),
#     		nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#     		nn.ReLU(True),
#     		nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#     		nn.ReLU(True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         ])
#         # if norm == 'VGG16':
#         #     self.features_1 = make_layers_vgg16(input_dim)

#         if init:
#             initialize_weights(self)

#     def forward(self, x):
#         for module in self.features_1:
#             x = module(x)
#         return x

#     def get_features(self, drop=0):
#         return nn.Sequential(*ml_drop(self.features_1, drop))

#     def get_conv3_features(self, x):
#         for i in range(10):
#             module = self.features_1[i]
#             x = module(x)
#         return x

#     def get_conv4_features(self, x):
#         for i in range(12):
#             module = self.features_1[i]
#             x = module(x)
#         return x

#     def get_convpp5_features(self, x):
#         for i in range(14):
#             module = self.features_1[i]
#             x = module(x)
#         return x

#     def get_conv5_features(self, x):
#         for module in self.features_1:
#             x = module(x)
#         return x

#     def get_net_info(self):
#         dict_info = super(VGG_Conv, self).get_net_info()
#         dict_info.update({'input_dim': self.input_dim, 'norm': self.norm})
#         return dict_info

#     def get_filters(self):
#         arr = self.get_features().state_dict()['0.weight'].cpu().numpy()
#         arr, mini, maxi = normalize(arr)
#         max_value = np.max([np.abs(mini), np.abs(maxi)])
#         arr = np.swapaxes(arr,1,2)
#         arr = np.swapaxes(arr,2,3)
#         input_dim = self.input_dim
#         ks = arr.shape[1]
#         ks1 = ks+1
#         # if self.norm == 'LRN' or self.norm == 'BN':
#         wi, hi = (12, 8)
#         # if self.norm == 'VGG16':
#         #     wi, hi = (8, 8)
#         if input_dim == 3:
#             final_img = np.ones((hi*ks1, wi*ks1,3))
#             for i in range(hi):
#                 for j in range(wi):
#                     final_img[ks1*i:ks1*i+ks, ks1*j:ks1*j+ks, :] = arr[i*wi + j]          
#         else:
#             final_img = np.ones((input_dim*ks1,wi*hi*ks1))
#             nf = int(input_dim/2)
#             for i in range(wi*hi):
#                 for j in range(nf):
#                     final_img[ks1*j:ks1*j+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j]
#                     final_img[ks1*(j+nf):ks1*(j+nf)+ks, ks1*i:ks1*i+ks] = arr[i,:,:,2*j+1]
#         return {'filter':final_img}

    # def get_features(self):
    #     return nn.Sequential(*self.features_1)

class VGG(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    layer_dict = {'pool2':8,'conv3':10, 'conv4':12, 'conv5':14, 'pool5':15, 'fc6':3, 'fc7':6, 'fc8':7}
    tile_filters = (12,8)
    def __init__(self, input_dim=3, output_dim=2, norm='BN', dropout=0.5, init=True):
        super(VGG, self).__init__()
        self.input_dim = input_dim
        self.norm = norm
        if norm == 'BN':
            norms = [nn.BatchNorm2d(96), nn.BatchNorm2d(256)]
        elif norm == 'LRN':
            norms = [LRN(), LRN()]
        elif norm == 'NLM':
            norms = [NLM(), NLM()]
        self.features_1 = nn.ModuleList([
            # 224x224
            nn.Conv2d(input_dim, 96, kernel_size=7, stride=2, padding=0),
            nn.ReLU(True),
            norms[0],
            # 109x109
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 55x55
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1),
            nn.ReLU(True),
            norms[1],
            # 28x28
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 14x14
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        self.classifier_1 = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim)
        ])
        if init:
            initialize_weights(self)

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features_1, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier_1, drop))

    def get_net_info(self):
        dict_info = super(VGG, self).get_net_info()
        dict_info.update({'input_dim': self.input_dim, 'norm': self.norm})
        return dict_info

    # def get_fcpp6_features(self, x):
    #     x = super(VGG, self).forward(x) 
    #     x = x.view(x.size(0), -1)
    #     for i in range(2):
    #         module = self.classifier_1[i]
    #         x = module(x)
    #     return x

    # def get_fc6_features(self, x):
    #     x = super(VGG, self).forward(x) 
    #     x = x.view(x.size(0), -1)
    #     for i in range(3):
    #         module = self.classifier_1[i]
    #         x = module(x)
    #     return x

    # def get_fc7_features(self, x):
    #     x = super(VGG, self).forward(x) 
    #     x = x.view(x.size(0), -1)
    #     for i in range(6):
    #         module = self.classifier_1[i]
    #         x = module(x)
    #     return x

    # def forward(self, x):
    #     x = super(VGG, self).forward(x) 
    #     x = x.view(x.size(0), -1)
    #     for module in self.classifier_1:
    #         x = module(x)
    #     return x

# Used to make vgg16 layers
class VGG_16(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    layer_dict = {'pool2':24,'conv3':26, 'conv4':28, 'conv5':30, 'pool5':31, 'fc6':3, 'fc7':6}
    tile_filters = (8,8)
    def __init__(self, input_dim=3, output_dim=2, dropout=0.5, init=True):
        super(VGG_16, self).__init__()
        self.input_dim = input_dim
        self.features = nn.ModuleList([
            # 224x224
            nn.Conv2d(self.input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 112x112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 56x56
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 28x28
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14x14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # 7x7
        ])

        self.classifier = nn.ModuleList([
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim)
        ])

        if init:
            initialize_weights(self)

    def get_features(self, drop=0):
        return nn.Sequential(*ml_drop(self.features, drop))

    def get_classifier(self, drop=0):
        return nn.Sequential(*ml_drop(self.classifier, drop))

    def get_net_info(self):
        dict_info = super(VGG_16, self).get_net_info()
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

    # def get_conv4_features(self, x):
    #     return self.feature_net.get_conv4_features(x)

    # def get_convpp5_features(self, x):
    #     return self.feature_net.get_convpp5_features(x)

    # def get_conv5_features(self, x):
    #     return self.feature_net.get_conv5_features(x)

    # def get_fc6_features(self, x):
    #     return self.feature_net.get_fc6_features(x)

    # def get_fc7_features(self, x):
    #     return self.feature_net.get_fc7_features(x)



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
        self.decoder = decoder
        self.groups = 1
        if norm == 'caffe':
            raise Exception('Caffe is deprecated, inconsequential use of BN')
        if self.norm == 'alex':
            arch = AlexNet
        elif self.norm == 'caffe':
            arch = CaffeNet
        elif self.norm == 'caffe_bn':
            arch = CaffeNet_BN
        elif self.norm == 'caffe_bn_g2':
            arch = CaffeNet_BN
            self.groups = 2
        elif self.norm == 'caffe_g2':
            arch = CaffeNet_NLM
            self.groups = 2
        elif self.norm == 'caffe_nlm':
            arch = CaffeNet_NLM
        elif self.norm == 'vgg':
            arch = VGG_Imagenet
        elif self.norm == 'vgg16':
            arch = VGG_16
        elif self.norm == 'vgg16bn':
            arch = VGG_16_BN
        elif self.norm == 'caffe_vgg_bn':
            arch = CaffeNet_BN
        else:
            arch = VGG

        if 'rgb' in self.modalities:
            self.app_net = arch(input_dim=3, dropout=dropout, groups=self.groups, 
                leaky_relu=leaky_relu)
        if 'of' in self.modalities:
            if self.norm == 'caffe_vgg_bn':
                self.mot_net = VGG_16_BN(input_dim=num_frames*2, dropout=dropout)
            else:
                self.mot_net = arch(input_dim=num_frames*2, dropout=dropout, groups=self.groups,
                    leaky_relu=leaky_relu)
        if 'cod' in self.modalities:
            self.cod_net = arch(input_dim=num_frames_cod*3, dropout=dropout, groups=self.groups,
                leaky_relu=leaky_relu)
        if 'rgb2' in self.modalities:
            self.ap2_net = arch(input_dim=3, dropout=dropout, groups=self.groups,
                leaky_relu=leaky_relu)

    def get_app_net(self):
        return self.app_net

    def get_mot_net(self):
        return self.mot_net

    def get_cod_net(self):
        return self.cod_net

    def set_app_net(self, app_net):
        self.app_net = app_net

    def get_net_info(self):
        dict_info = super(Base_Siamese, self).get_net_info()
        dict_info.update({'norm': self.norm, 'num_frames': self.num_frames, 
            'num_frames_cod': self.num_frames_cod, 'modalities':self.modalities, 
            'decoder':self.decoder})
        return dict_info

    def get_filters(self):
        dict_filter = {}
        if 'rgb' in self.modalities:
            dict_filter['app'] = self.app_net.get_filters()['filter']
        if 'of' in self.modalities:
            dict_filter['mot'] = self.mot_net.get_filters()['filter']
        if 'cod' in self.modalities:
            dict_filter['cod'] = self.cod_net.get_filters()['filter']
        if 'rgb2' in self.modalities:
            dict_filter['ap2'] = self.ap2_net.get_filters()['filter']
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

            # nn.Dropout(dropout),
            # nn.Linear(256 * 6 * 6, 4096),
            # nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096,1000)


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
            if 'rgb2' in self.modalities:
                image = sample.pop(0)
                image_feat = self.ap2_net.get_feature_output(image, layer=self.layer)
                image_feat = image_feat.view(image_feat.size(0), -1)
                features.append(image_feat)
                nonzeros.append(image_feat.data.nonzero().size(0))

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
                # nonzeros.append(image_feat.data.nonzero().size(0))
                nonzeros.append(0)
            if 'of' in self.modalities:
                flow = sample.pop(0)
                flow_feat = self.mot_net.get_feature_output(flow, layer=self.layer)
                flow_feat = flow_feat.view(flow_feat.size(0), -1)
                features.append(flow_feat)
                lengths.append((flow_feat ** 2).sum(dim=1).sqrt())
                # nonzeros.append(flow_feat.data.nonzero().size(0))
                nonzeros.append(0)
            if 'cod' in self.modalities:
                cod = sample.pop(0)
                cod_feat = self.cod_net.get_feature_output(cod, layer=self.layer)
                cod_feat = cod_feat.view(cod_feat.size(0), -1)
                features.append(cod_feat)
                lengths.append((cod_feat ** 2).sum(dim=1).sqrt())
                # nonzeros.append(cod_feat.data.nonzero().size(0))
                nonzeros.append(0)
            if 'rgb2' in self.modalities:
                image = sample.pop(0)
                image_feat = self.ap2_net.get_feature_output(image, layer=self.layer)
                image_feat = image_feat.view(image_feat.size(0), -1)
                features.append(image_feat)
                lengths.append((image_feat ** 2).sum(dim=1).sqrt())
                # nonzeros.append(image_feat.data.nonzero().size(0))
                nonzeros.append(0)

        # feat_mean = 0
        # nonzero = 0
        # counter = 0
        # size = 0
        # for feat in features:
        #     size = feat.size(0) * feat.size(1)
        #     feat_mean += feat.data.mean()
        #     nonzero += torch.nonzero(feat.data).size(0)
        #     counter += 1
        # feat_mean /= counter
        # nonzero /= (counter * size)

        # print(float(feat_mean), float(nonzero))


        if len(self.modalities) == 2:
            s = features[0].size(1)
            if self.union:
                s = int(s / 2)
            sim_true_1 = self.sim_func(features[0][:,:s], features[1][:,:s])      #   1
            sim_true_2 = self.sim_func(features[2][:,:s], features[3][:,:s])      #   1
            sim_false_1 = self.sim_func(features[0][:,:s], features[2][:,:s]) #   2
            sim_false_2 = self.sim_func(features[1][:,:s], features[3][:,:s]) #   2                  
            final_sim_true_1 = (sim_true_1 + sim_true_2) / 2
            final_sim_false_1 = (sim_false_1 + sim_false_2) / 2

            return sim_true_1, sim_true_2, sim_false_1, sim_false_2, nonzeros

        if len(self.modalities) == 3:
            s = features[0].size(1)
            if self.union:
                s = int(s / 2)
            sim_true_1 = self.sim_func(features[0][:,:s], features[1][:,:s])    #   1
            sim_true_2 = self.sim_func(features[0][:,-s:], features[2][:,:s])   #   2
            sim_true_3 = self.sim_func(features[1][:,-s:], features[2][:,-s:])  #   3
            sim_true_4 = self.sim_func(features[3][:,:s], features[4][:,:s])    #   1
            sim_true_5 = self.sim_func(features[3][:,-s:], features[5][:,:s])   #   2
            sim_true_6 = self.sim_func(features[4][:,-s:], features[5][:,-s:])  #   3

            sim_false_1 = self.sim_func(features[0][:,:s], features[4][:,:s])   #   4
            sim_false_2 = self.sim_func(features[0][:,-s:], features[5][:,:s])  #   5
            sim_false_3 = self.sim_func(features[1][:,:s], features[3][:,:s])   #   4
            sim_false_4 = self.sim_func(features[1][:,-s:], features[5][:,-s:]) #   6
            sim_false_5 = self.sim_func(features[2][:,:s], features[3][:,-s:])  #   5
            sim_false_6 = self.sim_func(features[2][:,-s:], features[4][:,-s:]) #   6

            final_sim_true_1 = (sim_true_1 + sim_true_4) / 2    # rgb - of
            final_sim_true_2 = (sim_true_2 + sim_true_5) / 2    # rgb - cod
            final_sim_true_3 = (sim_true_3 + sim_true_6) / 2    # of - cod

            final_sim_false_1 = (sim_false_1 + sim_false_3) / 2 # rgb - of
            final_sim_false_2 = (sim_false_2 + sim_false_5) / 2 # rgb - cod
            final_sim_false_3 = (sim_false_4 + sim_false_6) / 2 # of - cod
            
            return final_sim_true_1, final_sim_true_2, final_sim_true_3, \
            final_sim_false_1, final_sim_false_2, final_sim_false_3


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
        if self.similarity_scheme == 'linear':
            self.sim_func = Sim_func(lin_sim, self.eps)
        if self.similarity_scheme == 'own':
            self.sim_func = Sim_func(own_sim, self.eps)
        if self.similarity_scheme == 'euclidean_nonorm':
            self.sim_func = Sim_func(euc_nonorm_sim, self.eps)

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
            if 'rgb2' in self.modalities:
                image = sample.pop(0)
                if 'rgb2' in dismissed:
                    image_feat = None
                else:
                    image_feat = self.ap2_net.get_feature_output(image, layer=layer)
                features.append(image_feat)
        return features


class Base_Single(Base_Network):
    input_spatial_size = (224, 224)
    input_dim = None
    def __init__(self, norm='BN', dropout=0.5):
        super(Base_Single, self).__init__()
        self.norm = norm
  
        if self.norm == 'alex':
            arch = AlexNet
        elif self.norm == 'caffe':
            arch = CaffeNet
        elif self.norm == 'caffe_bn':
            arch = CaffeNet_BN
        elif self.norm == 'vgg':
            arch = VGG_Imagenet
        elif self.norm == 'vgg16':
            arch = VGG_16
        else:
            arch = VGG

        self.app_net = arch(input_dim=3, dropout=dropout)

    def get_app_net(self):
        return self.app_net

    def get_net_info(self):
        dict_info = super(Base_Single, self).get_net_info()
        dict_info.update({'norm': self.norm})
        return dict_info

    def get_filters(self):
        dict_filter = {}    
        dict_filter['app'] = self.app_net.get_filters()['filter']
        return dict_filter

    def get_features(self, drop=0):
        raise NotImplementedError('previous implementation deprecated')
        # return self.app_net.get_features(drop=drop), self.mot_net.get_features(drop=drop)

class Single_def(Base_Single):
    def __init__(self, norm='BN', bottleneck=4096, early_cat=True, dropout=0.5, init=True,
        union=False):
        super(Single_def, self).__init__(norm=norm, dropout=dropout)
        if bottleneck != 4096:
            print('bottelneck deprecated!!!!!!!!!!!!!!')
        self.bottleneck = bottleneck
        self.early_cat = early_cat
        self.union = union
        size = 4096
        if self.union:
            size = 2048
        if self.early_cat:
            self.classifier_2 = nn.ModuleList([
                nn.Dropout(dropout),
                nn.Linear(size*3, 4096),
                nn.ReLU(True),
                nn.Linear(4096, 2)
            ])

        else:             
            self.classifier_2 = nn.ModuleList([
                nn.Linear(4096*3, 2),
            ])

        if init:
            initialize_weights(self)


    def forward(self, *sample):
        sample = list(sample)
        if self.early_cat:
            layer = 'fc6'
        else:
            layer = 'fc7'
        features = []
        for image in sample:
            image_feat = self.app_net.get_feature_output(image, layer=layer)
            image_feat = image_feat.view(image_feat.size(0), -1) 
            s = image_feat.size(1)
            if self.union:
                s = int(s / 2)
            features.append(image_feat[:,-s:])


        outputs = []
        outputs.append(torch.cat([features[1], features[2], features[3]], 1)) # T
        outputs.append(torch.cat([features[1], features[0], features[3]], 1)) # F
        outputs.append(torch.cat([features[1], features[4], features[3]], 1)) # F
        # outputs.append(torch.cat([features[1], features[5], features[3]], 1)) # F
        # outputs.append(torch.cat([features[3], features[2], features[1]], 1)) # T
        # outputs.append(torch.cat([features[3], features[0], features[1]], 1)) # F
        # outputs.append(torch.cat([features[3], features[4], features[1]], 1)) # F
        # outputs.append(torch.cat([features[3], features[5], features[1]], 1)) # F
        predictions = []
        for output in outputs:
            for module in self.classifier_2:
                output = module(output)
            predictions.append(output)
        return predictions

    def get_net_info(self):
        dict_info = super(Single_def, self).get_net_info()
        dict_info.update({'bottleneck': self.bottleneck, 'early_cat': self.early_cat})
        return dict_info

class Single_fm(Base_Single):
    def __init__(self, norm='BN', layer='fc6', dropout=0.5, scheme=0, similarity_scheme='cosine',
        eps=0.001):
        super(Single_fm, self).__init__(norm=norm, dropout=dropout)
        self.layer = layer
        self.scheme = scheme
        self.similarity_scheme = similarity_scheme
        self.eps = eps
        self._set_similarity_func()

    def forward(self, *sample):
        sample = list(sample)
        features = []
        for image in sample:
            image_feat = self.app_net.get_feature_output(image, layer=self.layer)
            image_feat = image_feat.view(image_feat.size(0), -1)
            features.append(image_feat)

        if self.scheme == 0:
            sim_true_1 = self.sim_func(features[1], features[2])  #   1
            sim_true_2 = self.sim_func(features[2], features[3])  #   1
            sim_false_1 = self.sim_func(features[0], features[3]) #   2
            sim_false_2 = self.sim_func(features[1], features[4]) #   2

        if self.scheme == 1:
            sim_true_1 = self.sim_func(features[2], features[1])  #   1
            sim_true_2 = self.sim_func(features[2], features[3])  #   1
            sim_false_1 = self.sim_func(features[2], features[0]) #   2
            sim_false_2 = self.sim_func(features[2], features[4]) #   2

        final_sim_true_1 = (sim_true_1 + sim_true_2) / 2
        final_sim_false_1 = (sim_false_1 + sim_false_2) / 2

        return final_sim_true_1, final_sim_false_1

    def set_layer(self, layer):
        self.layer = layer

    def get_net_info(self):
        dict_info = super(Single_fm, self).get_net_info()
        dict_info.update({'layer': self.layer})
        return dict_info

    def _set_similarity_func(self):
        if self.similarity_scheme == 'cosine':
            self.sim_func = Sim_func(cos_sim, self.eps)
        if self.similarity_scheme == 'euclidean':
            self.sim_func = Sim_func(euc_sim, self.eps)
        if self.similarity_scheme == 'linear':
            self.sim_func = Sim_func(lin_sim, self.eps)
        if self.similarity_scheme == 'own':
            self.sim_func = Sim_func(own_sim, self.eps)
        if self.similarity_scheme == 'euclidean_nonorm':
            self.sim_func = Sim_func(euc_nonorm_sim, self.eps)

class Container(nn.Module):
    input_spatial_size = None
    input_dim = None
    def __init__(self, net_1, net_2):
        super(Container, self).__init__()
        net_2.app_net = net_1.app_net
        self.net_1 = net_1
        self.net_2 = net_2

# Only for RGB images currently
class Two_Stream(nn.Module):
    input_spatial_size = (224, 224)
    input_dim = 3
    def __init__(self, net_1, net_2, dropout=0.5, fc7_layer=False):
        super(Two_Stream, self).__init__()
        self.net_1 = net_1
        self.net_2 = net_2
        self.fc7_layer = fc7_layer
        if self.fc7_layer:
            self.classifier = nn.Linear(4096*2, 4096)
        set_dropout(self, dropout)

    def forward(self, x):
        out_1 = self.net_1(x)
        out_2 = self.net_2(x)  
        return out_1, out_2      

            

class Caffe_biagio(nn.Module):

    def __init__(self, classes=1000, use_bn=False, groups = 2):
        super(Caffe_biagio, self).__init__()
        # print( 'GROUPS: ', groups)

        self.conv = nn.Sequential()
        self.conv.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        self.conv.add_module('relu1_s1',nn.ReLU(inplace=True))
        if use_bn: self.conv.add_module('bn1_s1',nn.BatchNorm2d(96))
        self.conv.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        # if not use_bn: self.conv.add_module('lrn1_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        # if not use_bn: self.conv.add_module('lrn1_s1',nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=groups))
        self.conv.add_module('relu2_s1',nn.ReLU(inplace=True))
        if use_bn: self.conv.add_module('bn2_s1',nn.BatchNorm2d(256))
        self.conv.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        # if not use_bn: self.conv.add_module('lrn2_s1',LRN(local_size=5, alpha=0.0001, beta=0.75))
        #if not use_bn: self.conv.add_module('lrn2_s1',nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75))

        self.conv.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        self.conv.add_module('relu3_s1',nn.ReLU(inplace=True))
        if use_bn: self.conv.add_module('bn3_s1',nn.BatchNorm2d(384))

        self.conv.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=groups))
        self.conv.add_module('relu4_s1',nn.ReLU(inplace=True))
        if use_bn: self.conv.add_module('bn4_s1',nn.BatchNorm2d(384))

        self.conv.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=groups))
        self.conv.add_module('relu5_s1',nn.ReLU(inplace=True))
        if use_bn: self.conv.add_module('bn5_s1',nn.BatchNorm2d(256))
        self.conv.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))

        # self.features = nn.Sequential()
        # self.features.add_module('conv1_s1',nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0))
        # self.features.add_module('bn1_s1',nn.BatchNorm2d(96))
        # self.features.add_module('relu1_s1',nn.ReLU(inplace=True))
        # self.features.add_module('pool1_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        #
        # self.features.add_module('conv2_s1',nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=groups))
        # self.features.add_module('bn2_s1',nn.BatchNorm2d(256))
        # self.features.add_module('relu2_s1',nn.ReLU(inplace=True))
        # self.features.add_module('pool2_s1',nn.MaxPool2d(kernel_size=3, stride=2))
        #
        # self.features.add_module('conv3_s1',nn.Conv2d(256, 384, kernel_size=3, padding=1))
        # self.features.add_module('bn3_s1',nn.BatchNorm2d(384))
        # self.features.add_module('relu3_s1',nn.ReLU(inplace=True))
        #
        # self.features.add_module('conv4_s1',nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=groups))
        # self.features.add_module('bn4_s1',nn.BatchNorm2d(384))
        # self.features.add_module('relu4_s1',nn.ReLU(inplace=True))
        #
        # self.features.add_module('conv5_s1',nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=groups))
        # self.features.add_module('bn5_s1',nn.BatchNorm2d(256))
        # self.features.add_module('relu5_s1',nn.ReLU(inplace=True))
        # self.features.add_module('pool5_s1',nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc6 = nn.Sequential()
        self.fc6.add_module('fc6_s1',nn.Linear(256*3*3, 1024))
        self.fc6.add_module('bn6_s1',nn.BatchNorm1d(1024))
        self.fc6.add_module('relu6_s1',nn.ReLU(inplace=True))
        #if use_bn: self.fc6.add_module('bn6_s1',nn.BatchNorm1d(1024))
        self.fc6.add_module('drop6_s1',nn.Dropout(p=0.50))

        self.fc7 = nn.Sequential()
        self.fc7.add_module('fc7',nn.Linear(9*1024,4096))
        self.fc7.add_module('bn7',nn.BatchNorm1d(4096))
        self.fc7.add_module('relu7',nn.ReLU(inplace=True))
        #if use_bn: self.fc7.add_module('bn7',nn.BatchNorm1d(4096))
        self.fc7.add_module('drop7',nn.Dropout(p=0.50))

        self.classifier = nn.Sequential()
        self.classifier.add_module('fc8',nn.Linear(4096, classes))

        #self.apply(weights_init)

    def load(self,checkpoint):
        model_dict = self.state_dict()
        #print [k for k, v in model_dict.items()]

        pretrained_dict = torch.load(checkpoint)
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        #print [k for k, v in pretrained_dict.items()]

        #for k, v in pretrained_dict.items():
            #pretrained_dict['main.'+k] = v

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict}# and 'fc8' not in k}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        # print [k for k, v in pretrained_dict.items()]

    def save(self,checkpoint):
        torch.save(self.state_dict(), checkpoint)

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)

        x_list = []
        for i in range(9):
            z = self.conv(x[i])
            z = self.fc6(z.view(B,-1))
            z = z.view([B,1,-1])
            x_list.append(z)

        x = cat(x_list,1)
        x = self.fc7(x.view(B,-1))
        x = self.classifier(x)

        return x


    # self.input_spatial_size = self.siamese.input_spatial_size
    # self.input_dim = self.siamese.input_dim

    # def forward(self, sample_1, sample_2):
    #     output_1 = self.siamese(*sample)
    #     output_2 =s




# class Net_ar(Base_Network):
#     input_spatial_size = None
#     input_dim = None
#     data_keys_dict = {'ucf':101, 'hmdb':51, 'act':43, 'olympic':16}
#     def __init__(self, feature_net, dropout=0.5, data_key='ucf'):
#         super(Base_Network, self).__init__()
#         if isinstance(feature_net, Base_Network):
#             self.feature_net = feature_net
#         else:
#             self.feature_net = get_network_dict(feature_net)
#         self.input_spatial_size = self.feature_net.input_spatial_size
#         self.input_dim = self.feature_net.input_dim
#         self.data_key = data_key
#         output_dim = self.data_keys_dict[data_key]
#         self.classifier_ar = nn.Linear(4096, output_dim)
#         for module in self.modules():
#             if isinstance(module, nn.Dropout):
#                 module.p = dropout

#     def forward(self, x):
#         x = self.feature_net.get_feature_output(x, layer='fc7')
#         return self.classifier_ar(x)

#     def get_features(self, drop=0):
#         return self.feature_net.get_features(drop=drop)

#     def get_classifier(self, drop=0):
#         if drop == 0 or drop == 7:
#             return nn.Sequential(self.feature_net.get_classifier(-1), self.classifier_ar)
#         else:
#             return nn.Sequential(self.feature_net.get_classifier(drop))

#     def get_net_info(self):
#         dict_info = super(Net_ar, self).get_net_info()
#         dict_info.update({'feature_net': self.feature_net.get_net_info(), 'data_key': self.data_key})
#         return dict_info

#     def get_filters(self):
#         return self.feature_net.get_filters()

#     def get_feature_output(self, x, layer='fc8'):
#         if layer == 'fc8':
#             x = self.feature_net.get_feature_output(x, layer='fc7')
#             return self.classifier_ar(x) 
#         else:
#             return self.feature_net.get_feature_output(x, layer)

#     def cod_to_coi(self):
#         super(Net_ar, self).cod_to_coi()
#         self.feature_net.input_dim += 3

#     def freeze_layers(self, layer='input'):
#         if layer == 'fc8':
#             for param in self.parameters():
#                 param.requires_grad = False
#         else:
#             self.feature_net.freeze_layers(layer)

#     def reset_fc7(self):
#         self.feature_net.reset_fc7()
        # image = self.app_net.get_feature_output(image, layer=self.layer)
        # flow_1 = self.mot_net.get_feature_output(flow_1, layer=self.layer)
        # flow_2 = self.mot_net.get_feature_output(flow_2, layer=self.layer)
        # image = image.view(image.size(0), -1)
        # flow_1 = flow_1.view(flow_1.size(0), -1)
        # flow_2 = flow_2.view(flow_2.size(0), -1)

        # if self.output_dim == -2:
        #     image = self.app_net.get_conv5_features(image)
        #     flow_1 = self.mot_net.get_conv5_features(flow_1)
        #     flow_2 = self.mot_net.get_conv5_features(flow_2)
        #     image = image.view(image.size(0), -1)
        #     flow_1 = flow_1.view(flow_1.size(0), -1)
        #     flow_2 = flow_2.view(flow_2.size(0), -1)
        # elif self.output_dim == -1.5:
        #     image = self.app_net.get_fcpp6_features(image)
        #     flow_1 = self.mot_net.get_fcpp6_features(flow_1)
        #     flow_2 = self.mot_net.get_fcpp6_features(flow_2)
        # elif self.output_dim == -1:
        #     image = self.app_net.get_fc6_features(image)
        #     flow_1 = self.mot_net.get_fc6_features(flow_1)
        #     flow_2 = self.mot_net.get_fc6_features(flow_2)
        # elif self.output_dim == 0:
        #     image = self.app_net.get_fc7_features(image)
        #     flow_1 = self.mot_net.get_fc7_features(flow_1)
        #     flow_2 = self.mot_net.get_fc7_features(flow_2)
        # if self.output_dim != -1:
        #     for module in self.app_classifier:
        #         image = module(image)
        #     for module in self.mot_classifier:
        #         flow_1 = module(flow_1)
        #     for module in self.mot_classifier:
        #         flow_2 = module(flow_2)

        # if self.variational != 0:
        #     random = Variable(image.data.clone().normal_().cuda())
        #     image_var = image + random
        # else:
        #     image_var = image






# dict_info = {'type':VGG, 'input_dim':5}
# net = get_network(dict_info)
# print(net.get_setup())
# Type = type(net)
# net = Type() 
# print(net.get_setup())





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


