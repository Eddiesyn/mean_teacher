import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import time

import config
from utils import *

def prepare_mnist(root, transform):
    # Don't use drop last and shuffle
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)

    return train_dataset, val_dataset

def prepare_cifar10(root, mean, std):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_val)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_dataset, val_dataset, classes

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

if __name__ == '__main__':
    # writer = SummaryWriter(log_dir='./results')
    train_batch_logger = Logger('./results/train_batch.log', ['epoch', 'batch', 'iter',
                                                              'ce_loss', 'cons_loss', 'prec1',
                                                              'prec5', 'ema_prec1', 'ema_prec5', 'lr'])
    train_epoch_logger = Logger('./results/train_epoch.log', ['epoch', 'ce_loss', 'cons_loss',
                                                              'prec1', 'prec5', 'ema_prec1', 'ema_prec5'])
    cfg = vars(config)
    global_step = 0

    # Meter preparation
    time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    loss_meter = AverageMeter()
    CE_loss_meter = AverageMeter()
    Consistency_loss_meter = AverageMeter()
    ema_CE_loss_meter = AverageMeter()
    weight_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    ema_top1 = AverageMeter()
    ema_top5 = AverageMeter()

    # use a resnet18 model
    model = create_model() # student
    ema_model = create_model(ema=True) # teacher

    train_set, val_set, classes = prepare_cifar10(cfg['dataset_root'], cfg['mean'], cfg['std'])
    train_loader, val_loader = sample_train(train_set, val_set, cfg['batch_size'],
                                            cfg['NUM_LABELS'], len(classes), cfg['seed'], cfg['ratio'])

    # classification error is ignored for unlabeled samples, but averaged by whole batch, not just labeled samples
    class_criterion = nn.CrossEntropyLoss(ignore_index=config['NO_LABEL'], reduction='sum').cuda()
    if cfg['consistency_type'] == 'mse':
        consistency_criterion = softmax_mse_loss
    elif cfg['consistency_type'] == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        consistency_criterion = None
        exit('wrong consistency type! Check config file!')

    optimizer = torch.optim.SGD(model.parameters(), cfg['init_lr'],
                                momentum=0.99, weight_decay=cfg['weight_decay'])

    for epoch in range(cfg['num_epochs']):
        model.train()
        ema_model.train()

        end_time = time.time()
        for step, ((input, ema_input), target) in enumerate(train_loader):
            time_meter.update(time.time() - end_time)
            # adjust_learning_rate(optimizer, epoch, step)

            input = input.cuda()
            ema_input = ema_input.cuda()
            target = target.cuda()

            labeled_batch_size = target.data.ne(config['NO_LABEL']).sum()
            minibatch_size = target.size(0)
            print('labeled batch size {}/{}'.format(labeled_batch_size, minibatch_size))

            ema_out = ema_model(ema_input)
            out = model(input)

            class_loss = class_criterion(out, target) / minibatch_size
            CE_loss_meter.update(class_loss.item(), minibatch_size)

            # check ema classification loss
            ema_class_loss = class_criterion(ema_out, target) / minibatch_size
            ema_CE_loss_meter.update(ema_class_loss.item(), minibatch_size)

            # get consistency loss
            consistency_weight = get_current_consistency_weight(epoch, cfg['consistency_weight'], cfg['rampup_length'])
            weight_meter.update(consistency_weight)
            consistency_loss = consistency_criterion(out, ema_out) * consistency_weight / minibatch_size
            Consistency_loss_meter.update(consistency_loss, minibatch_size)

            loss = class_loss + consistency_loss
            assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion {}'.format(loss.item())
            loss_meter.update(loss.item(), minibatch_size)

            prec1, prec5 = accuracy(out.detach(), target.detach(), topk=(1, 5))
            top1.update(prec1, minibatch_size)
            top5.update(prec5, minibatch_size)

            ema_prec1, ema_prec5 = accuracy(ema_out.detach(), target.detach(), topk=(1, 5))
            ema_top1.update(ema_prec1, minibatch_size)
            ema_top5.update(ema_prec5, minibatch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            update_ema_variable(model, ema_model, cfg['ema_decay'], global_step)

            batch_time_meter.update(time.time() - end_time)
            end_time = time.time()

            train_batch_logger.log([epoch, step, global_step, class_loss.item(),
                                    consistency_loss.item(), prec1, prec5,
                                    ema_prec1, ema_prec5, optimizer.param_group[0]['lr']])

            if step % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Class_loss {c_loss.val:.4f} ({c_loss.avg:.4f})\t'
                      'Cons_loss {cons_loss.val:.4f} ({cons_loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                      'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                    epoch,
                    step,
                    len(train_loader),
                    batch_time=batch_time_meter,
                    data_time=time_meter,
                    c_loss=CE_loss_meter,
                    cons_loss = Consistency_loss_meter,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[0]['lr']))





