import torch
from torch import nn

from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet
from torchsummary import summary


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()
    # model.named_parameters()返回模块参数的迭代器，产生参数名称和参数本身
    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnet2p1d', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      # n_classes=opt.n_finetune_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'resnet2p1d':
        model = resnet2p1d.generate_model(model_depth=opt.model_depth,
                                          n_classes=opt.n_classes,
                                          n_input_channels=opt.n_input_channels,
                                          shortcut_type=opt.resnet_shortcut,
                                          conv1_t_size=opt.conv1_t_size,
                                          conv1_t_stride=opt.conv1_t_stride,
                                          no_max_pool=opt.no_max_pool,
                                          widen_factor=opt.resnet_widen_factor)
    elif opt.model == 'wideresnet':
        model = wide_resnet.generate_model(
            model_depth=opt.model_depth,
            k=opt.wide_resnet_k,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'resnext':
        model = resnext.generate_model(model_depth=opt.model_depth,
                                       cardinality=opt.resnext_cardinality,
                                       n_classes=opt.n_classes,
                                       n_input_channels=opt.n_input_channels,
                                       shortcut_type=opt.resnet_shortcut,
                                       conv1_t_size=opt.conv1_t_size,
                                       conv1_t_stride=opt.conv1_t_stride,
                                       no_max_pool=opt.no_max_pool)
    elif opt.model == 'preresnet':
        model = pre_act_resnet.generate_model(
            model_depth=opt.model_depth,
            n_classes=opt.n_classes,
            n_input_channels=opt.n_input_channels,
            shortcut_type=opt.resnet_shortcut,
            conv1_t_size=opt.conv1_t_size,
            conv1_t_stride=opt.conv1_t_stride,
            no_max_pool=opt.no_max_pool)
    elif opt.model == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)

    return model


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')
        # pretrain = torch.load(pretrain_path, map_location='cuda')
        model = model.to('cuda')

        # print(summary(model,(3,16,112,112),batch_size=128))

        # model.load_state_dict(pretrain['state_dict'])
        # for k,v in pretrain['state_dict'].items():
        #     print('key:')
        #     print(k)
        #     print('value:')
        #     print(v)
        # print({k.replace('module.',''):v for k,v in pretrain['state_dict'].items()})
        model.load_state_dict({k.replace('module.',''):v for k,v in pretrain['state_dict'].items()}, False)
        tmp_model = model
        if model_name == 'densenet':
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)
        else:
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
