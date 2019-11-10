import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import os
import json

import opts
from utils import *
from train import train_epoch, validate_epoch
from dataset import prepare_cifar10
from mean import get_mean, get_std


def create_model(ema=False, num_classes=10):
    """Create a model for mean-teacher

    :param ema: if ema, the model is for saving EMA weights, teacher model
    :param num_classes:
    :return:
    """
    model = models.resnet18(pretrained=False)
    num_f = model.fc.in_features
    model.fc = nn.Linear(num_f, num_classes)
    model = model.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()

    return model

def create_3d_resnet(ema=False, num_classes=101):
    


if __name__ == '__main__':
    args = opts.parse_opts()
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    # for key in cfg.keys():
    #     print('{}: {}'.format(key, cfg[key]))
    # if not os.path.exists(os.path.join(args.result_path, 'config.py')):
    #     shutil.copyfile('./config.py', os.path.join(args.result_path, 'config.py'))
    args.scales = [args.initial_scale]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scales_step)
    args.arch = 'resnet18'
    args.mean = get_mean(1, dataset='activitynet')
    args.std = get_std(args.norm_value)

    print(args)
    with open(os.path.join(args.result_path, 'args.json'), 'w') as args_file:
        json.dump(vars(args), args_file)

    torch.manual_seed(args.manual_seed)

    # writer = SummaryWriter(log_dir='./results')
    train_batch_logger = Logger(os.path.join(args.result_path, args.pth_name + '_' + 'train_batch.log'),
                                ['epoch', 'batch', 'iter', 'class_loss', 'consistency_loss', 'prec1', 'ema_prec1', 'lr'])
    train_epoch_logger = Logger(os.path.join(args.result_path, args.pth_name + '_' + 'train_epoch.log'),
                                ['epoch', 'class_loss', 'consistency_loss', 'prec1', 'ema_prec1'])
    val_logger = Logger(os.path.join(args.result_path, args.pth_name + '_' + 'val.log'), ['epoch', 'loss', 'prec1'])

    student_model = create_model().cuda()  # student
    ema_model = create_model(ema=True).cuda()  # teacher

    train_set, val_set, classes = prepare_cifar10(args.dataset_root)
    train_loader, val_loader = sample_train(train_set, val_set, len(classes), args)

    # classification error is ignored for unlabeled samples, but averaged by whole batch, not just labeled samples
    class_criterion = nn.CrossEntropyLoss(ignore_index=args.NO_LABEL, reduction='sum').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        consistency_criterion = None
        exit('wrong consistency type! Check config file!')

    criterion = {'classification': class_criterion, 'consistency': consistency_criterion}

    optimizer = torch.optim.SGD(student_model.parameters(), args.init_lr,
                                momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    best_prec1 = 0
    for epoch in range(args.num_epochs):
        train_epoch(epoch, student_model, ema_model, train_loader, optimizer, criterion,
                    train_batch_logger, train_epoch_logger, args)

        state = {'epoch': epoch, 'state_dict': student_model.state_dict(), 'ema_state_dict': ema_model.state_dict(),
                 'optimizer': optimizer.state_dict(), 'best_prec1': best_prec1}
        save_checkpoint(state, False, args.result_path, args.pth_name)

        validation_loss, prec1 = validate_epoch(epoch, student_model, val_loader, criterion, val_logger, args)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        state = {'epoch': epoch, 'state_dict': student_model.state_dict(), 'ema_state_dict': ema_model.state_dict(),
                 'best_prec1': best_prec1, 'optimizer': optimizer.state_dict()}
        save_checkpoint(state, is_best, args.result_path, args.pth_name)
