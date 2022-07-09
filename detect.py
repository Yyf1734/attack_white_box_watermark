from copy import deepcopy
import json
from passport_attack_1 import test
import torch
import torch.nn as nn
from models.resnet_passport import ResNet18Passport
from models.resnet_passport_private import ResNet18Private
from experiments.utils import construct_passport_kwargs_from_dict
from dataset import prepare_dataset
from models.losses.sign_loss import SignLoss
import binascii
from models.layers.passportconv2d_private import PassportPrivateBlock

def sign_flip_attack(model, layer_name, s, first_layer = False):
    ex = 'layer' if first_layer is True else '%'
    for name, m in model.named_parameters():
        if 'classifier' not in name and 'bias' not in name and layer_name in name and ex not in name:
            print('flip ', name)
            if len(m.data.shape) == 1:
                m.data = m.data*s 
            else:
                m.data = m.data*(s.reshape(-1,1,1,1)) 

def layer_shuffle_attack(model, layer_in, layer_out, p, first_layer = False):
    ex = 'layer' if first_layer is True else '%'
    for name, m in model.named_parameters():
        if layer_in in name and 'fc' not in name and ex not in name:
            print('shuffle in ', name)
            m.data = m.data[p]
        if layer_out in name and 'weight' in name and 'bn.weight' not in name:
            print('shuffle out ',name)
            m.data = m.data[:,p]
    
# only scaling the conv weight (gn will recover it)
def _layer_scale_attack(model, layer_in, layer_out, scale_matrix_in, first_layer = False):
    ex = 'layer' if first_layer is True else '%'
    scale_matrix_out = 1.0/scale_matrix_in
    for name, m in model.named_parameters():
        if layer_in in name and 'fc' not in name and ex not in name and 'weight' in name and '.bn.' not in name:
            print('scale in', name)
            m.data = m.data*(scale_matrix_in if len(m.data.shape) == 1 else scale_matrix_in.reshape(-1, 1, 1, 1))

# scaling the conv weight, scale/bias parameter to scale the activation (thus the key_private need to be scaled)
def layer_scale_attack(model, layer_in, layer_out, scale_matrix_in, first_layer = False):
    ex = 'layer' if first_layer is True else '%'
    scale_matrix_out = 1.0/scale_matrix_in
    for name, m in model.named_parameters():
        if layer_in in name and 'fc' not in name and ex not in name:
            print('scale in', name)
            m.data = m.data*(scale_matrix_in if len(m.data.shape) == 1 else scale_matrix_in.reshape(-1, 1, 1, 1))
        if layer_out in name and 'weight' in name and 'bn.weight' not in name:
            print('scale out ',name)
            m.data = m.data*((scale_matrix_out.reshape(1,-1,1,1)) if len(m.data.shape) == 4 else scale_matrix_out.reshape(1, -1))
            
def main(nnum=16):
    import argparse

    parser = argparse.ArgumentParser(description='detect')
    parser.add_argument('--attack-rep', default=1, type=int)
    # model type
    parser.add_argument('--arch', default='resnet18', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100', 'imagenet1000'])
    parser.add_argument('--scheme', default=2, choices=[1, 2, 3], type=int)
    # model path
    parser.add_argument('--loadpath', default='logs/resnet_cifar100_v2/3/models/best.pth', help='path to model to be detected')
    # passport path
    parser.add_argument('--passport-config', default='passport_configs/resnet18_passport.json', help='path to passport config')
    parser.add_argument('--tagnum', default=torch.randint(100000, ()).item(), type=int,
                        help='tag number of the experiment')
    args = parser.parse_args()

    device = torch.device('cuda')

    passport_kwargs, plkeys = construct_passport_kwargs_from_dict({'passport_config': json.load(open(args.passport_config)),
                                                                   'norm_type': 'gn',
                                                                   'sl_ratio': 0.1,
                                                                   'key_type': 'shuffle'}, # shuffle
                                                                  True)
    nclass = {
        'cifar100': 100,
        'imagenet1000': 1000
    }.get(args.dataset, 10)

    if args.arch == 'alexnet':
        if args.scheme == 1:
            model = AlexNetPassport(inchan, nclass, passport_kwargs)
        else:
            model = AlexNetPassportPrivate(inchan, nclass, passport_kwargs)
    else:
        if args.scheme == 1:
            model = ResNet18Passport(num_classes=nclass, passport_kwargs=passport_kwargs)
        else:
            model = ResNet18Private(num_classes=nclass, passport_kwargs=passport_kwargs)

    sd = torch.load(args.loadpath)
    model.load_state_dict(sd)
    model = model.to(device)
    
    print("---------------------------parameters-----------------------------")
    names = ['layer3.0.convbnrelu_1.conv.weight', 'layer4.0.shortcut.weight']
    origin_weight = []
    for name, param in model.named_parameters():
        if name in names:
            origin_weight.append(param.data.detach())
        print(name, param.shape)
    print("---------------------------buffers-----------------------------")
    for name, param in model.named_buffers():
        print(name, param.shape)
    print(model)
#     exit()


    criterion = nn.CrossEntropyLoss()
    batch_size = 64

    trainloader, valloader = prepare_dataset({'transfer_learning': False,
                                              'dataset': args.dataset,
                                              'tl_dataset': '',
                                              'batch_size': batch_size,
                                              'shuffle_val': True})

    # 1 for passport, 2 for private
    print("#########################################################")
    print("##################   Before Attack   ##################")
    print("#########################################################")

    import numpy as np
    save_dict = {}

    print("################## For public passport ##################")
    scheme = 1
    # test performance
    valres = test(model, criterion, valloader, device, 1 if scheme != 1 else 0)
    
    print("################## For private passport ##################")
    scheme = 2
    # test performance
    valres = test(model, criterion, valloader, device, 1 if scheme != 1 else 0)

    nnum = nnum # 8
    import numpy as np
    neuron_index = torch.randperm(16)[:nnum]
    neuron_mask = torch.Tensor([0]*16)
    for ind in neuron_index:
        neuron_mask[ind] = 1
    
    neuron_index2 = torch.randperm(32)[:2*nnum]
    neuron_mask2 = torch.Tensor([0]*32)
    for ind in neuron_index2:
        neuron_mask2[ind] = 1
    
#     ## Flip Attack: \delta BER is proportional to the filp ratio
    cnt = 0
    total=0
    import numpy as np
    for name, param in model.named_parameters():
        if 'bn.weight' in name:
    #                 print(name, param.view(-1))
            cnt = cnt + (param.data < 0).sum().item()
            total = total + np.prod(param.data.shape)
    print('\n #############################  ',cnt,'/',total,'  #############################')
    s_512 = torch.sign(torch.randn(512)).cuda()
    s_flip = torch.Tensor([-1]*16)*neuron_mask
    for i in range(16):
        if s_flip[i] == 0:
            s_flip[i] = 1
    s_512 = torch.cat([torch.Tensor(s_flip) for i in range(512//16)]).cuda()
    sign_flip_attack(model, 'layer4.0.convbnrelu_1',s_512)
    sign_flip_attack(model, 'layer4.0.convbn_2',s_512)
    sign_flip_attack(model, 'layer4.0.shortcut',s_512)
    sign_flip_attack(model, 'layer4.1.convbnrelu_1',s_512)
    sign_flip_attack(model, 'layer4.1.convbn_2',s_512)

            
    # Shuffle Attack: 
    
    p_64 = torch.randperm(64)
    p_128 = torch.randperm(128)
    p_256 = torch.randperm(256)
    p_512 = torch.randperm(512) 
    p_256 = torch.cat([torch.randperm(16)+16*i for i in range(256//16)])
    p_512 = torch.cat([torch.randperm(16)+16*i for i in range(512//16)])
    
    
    p_in = torch.Tensor(list(range(16)))
    pos=0
    for i in range(16):
        if neuron_mask[i] == 1:
            p_in[i]=neuron_index[pos]
            pos = pos+1
    p_in = p_in.long()
    print(p_in)
    p_512 = torch.cat([p_in+16*i for i in range(512//16)])
    p_256 = torch.cat([p_in+16*i for i in range(256//16)])
    
    print('can keep shuffing by groups')
    layer_shuffle_attack(model, 'layer3.0.shortcut', '%',p_256)
    layer_shuffle_attack(model, 'layer3.0.convbnrelu_1', 'layer3.0.convbn_2',p_256) # Free
    layer_shuffle_attack(model, 'layer3.0.convbn_2', 'layer3.1.convbnrelu_1',p_256)
    layer_shuffle_attack(model, 'layer3.1.convbnrelu_1', 'layer3.1.convbn_2',p_256) # Free
    layer_shuffle_attack(model, 'layer3.1.convbn_2', 'layer4.0.convbnrelu_1',p_256)
    layer_shuffle_attack(model, '%', 'layer4.0.shortcut',p_256)
    
    layer_shuffle_attack(model, 'layer4.0.shortcut', '%',p_512)
    layer_shuffle_attack(model, 'layer4.0.convbnrelu_1', 'layer4.0.convbn_2',p_512) # Free
    layer_shuffle_attack(model, 'layer4.0.convbn_2', 'layer4.1.convbnrelu_1',p_512)
    layer_shuffle_attack(model, 'layer4.1.convbnrelu_1', 'layer4.1.convbn_2',p_512) # Free
    layer_shuffle_attack(model, 'layer4.1.convbn_2', 'linear.weight',p_512)
    
    ## Scaling Attack: 
    import numpy as np
    np.random.seed(100)
    power_2 = np.power(2,np.array(range(1, 53))) # np.arange(2, 20) # 
    power_1_2 = 1/power_2
    power_2 = np.concatenate((power_2, power_1_2))

    scale_pool_16 = 1/torch.from_numpy(np.random.choice(power_2, 16))*neuron_mask
    scale_pool_32 = 1/torch.from_numpy(np.random.choice(power_2, 32))*neuron_mask2
    for i in range(16):
        if scale_pool_16[i] == 0:
            scale_pool_16[i] = 1
    for i in range(32):
        if scale_pool_32[i] == 0:
            scale_pool_32[i] = 1
    
    scale_matrix_in_256 = scale_pool_16.repeat(16,1).T.reshape(-1).float().cuda()
    scale_matrix_in_512 = scale_pool_32.repeat(16,1).T.reshape(-1).float().cuda()

    layer_scale_attack(model, 'layer3.0.shortcut', '%',scale_matrix_in_256)
    layer_scale_attack(model, 'layer3.0.convbnrelu_1', 'layer3.0.convbn_2',scale_matrix_in_256) # Free
    layer_scale_attack(model, 'layer3.0.convbn_2', 'layer3.1.convbnrelu_1',scale_matrix_in_256)
    layer_scale_attack(model, 'layer3.1.convbnrelu_1', 'layer3.1.convbn_2',scale_matrix_in_256) # Free
    layer_scale_attack(model, 'layer3.1.convbn_2', 'layer4.0.convbnrelu_1',scale_matrix_in_256)
    layer_scale_attack(model, '%', 'layer4.0.shortcut',scale_matrix_in_256)
    
    print(scale_matrix_in_512)
    layer_scale_attack(model, 'layer4.0.shortcut', '%',scale_matrix_in_512)
    layer_scale_attack(model, 'layer4.0.convbnrelu_1', 'layer4.0.convbn_2',scale_matrix_in_512) # Free
    layer_scale_attack(model, 'layer4.0.convbn_2', 'layer4.1.convbnrelu_1',scale_matrix_in_512)
    layer_scale_attack(model, 'layer4.1.convbnrelu_1', 'layer4.1.convbn_2',scale_matrix_in_512) # Free
    layer_scale_attack(model, 'layer4.1.convbn_2', 'linear.weight',scale_matrix_in_512)

    #### Prune
    prune_ratio = 0.1
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name and 'bn.' not in name:
            print('prune ', name)
            weights=param.data.detach().cpu().numpy()
            weightshape=weights.shape
            rankedweights=np.abs(weights).reshape(weights.size).argsort()
            
            num = weights.size
            prune_num = int(np.round(num*prune_ratio))
            count=0
            masks = np.zeros_like(rankedweights)
            for n, rankedweight in enumerate(rankedweights):
                if rankedweight > prune_num:
                    masks[n]=1
                else: count+=1
            print("total weights:", num)
            print("weights pruned:",count)
            masks=masks.reshape(weightshape)
            weights=masks*weights
            # print(param.data.view(-1)[:10],  torch.from_numpy(weights).view(-1)[:10])
            # exit()
            param.data = torch.from_numpy(weights).to(dtype=torch.float32).cuda()

    names = ['layer3.0.shortcut', 'layer4.0.shortcut']
    print(names)

    for name, m in model.named_parameters():
        coff_list=[]
        coff_list_2 = []
        if 'layer3.0.convbnrelu_1.conv.weight' in name:
            (a,b,c,d) = m.data.size()
            for i in range(a):
                U10, sig10, V10 = np.linalg.svd(origin_weight[0].cpu().numpy()[i,:,:,:].reshape(b,c*d))
                U50, sig50, V50 = np.linalg.svd(m.data.detach().cpu().numpy()[i,:,:,:].reshape(b,c*d))
                coff = np.average(sig50/sig10)
                coff_list.append(coff)
            coff_list = 1/torch.as_tensor(coff_list).cuda()
            layer_scale_attack(model, 'layer3.0.shortcut', '%', coff_list)
            layer_scale_attack(model, 'layer3.0.convbnrelu_1', 'layer3.0.convbn_2', coff_list) # Free
            layer_scale_attack(model, 'layer3.0.convbn_2', 'layer3.1.convbnrelu_1', coff_list)
            layer_scale_attack(model, 'layer3.1.convbnrelu_1', 'layer3.1.convbn_2', coff_list) # Free
            layer_scale_attack(model, 'layer3.1.convbn_2', 'layer4.0.convbnrelu_1', coff_list)
            layer_scale_attack(model, '%', 'layer4.0.shortcut', coff_list)
    
        if 'layer4.0.shortcut.weight' in name:
            (a,b,c,d) = m.data.size()
            for i in range(a):
                U10, sig10, V10 = np.linalg.svd(origin_weight[1].cpu().numpy()[i,:,:,:].reshape(b,c*d))
                U50, sig50, V50 = np.linalg.svd(m.data.detach().cpu().numpy()[i,:,:,:].reshape(b,c*d))
                coff = np.average(sig50/sig10)
                coff_list_2.append(coff)
            coff_list_2 = 1/torch.as_tensor(coff_list_2).cuda()
            layer_scale_attack(model, 'layer4.0.shortcut', '%',coff_list_2)
            layer_scale_attack(model, 'layer4.0.convbnrelu_1', 'layer4.0.convbn_2',coff_list_2)
            layer_scale_attack(model, 'layer4.0.convbn_2', 'layer4.1.convbnrelu_1',coff_list_2)
            layer_scale_attack(model, 'layer4.1.convbnrelu_1', 'layer4.1.convbn_2',coff_list_2)
            layer_scale_attack(model, 'layer4.1.convbn_2', 'linear.weight',coff_list_2)
    
            
    # 1 for passport, 2 for private
    print("#########################################################")
    print("##################    After Attack   ##################")
    print("#########################################################")
    
    for m in model.modules():
        if isinstance(m, PassportPrivateBlock):
            print(m.get_scale().view(-1)[:10])
            print(m.get_bias().view(-1)[:10])
            break
   
    print("################## For public passport ##################")
    scheme = 1
    # test performance
    valres = test(model, criterion, valloader, device, 1 if scheme != 1 else 0)
    print("################## For private passport ##################")
    scheme = 2
    # test performance
    valres = test(model, criterion, valloader, device, 1 if scheme != 1 else 0)
    # output b
    for name, m in model.named_modules():
        if isinstance(m, SignLoss):
            b_raw = m.get_b()
            # print(b_raw)
            b_str = ''
            for i in range(len(b_raw) // 8):
                ii = ''
                for j in b_raw[i*8:i*8+8]:
                    ii += '1' if j > 0 else '0'
                b_str += chr(int(ii, 2))
            print('output string of b')
            print(name, ' -> ', b_str)
    return valres['signacc'], valres['acc']

if __name__ == '__main__':
    res = []
    res_signature = []
    ber, sig = main()
    res.append(ber)
    res_signature.append(sig)
    print(res)
    print(res_signature)
