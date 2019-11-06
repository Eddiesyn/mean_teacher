import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
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
    cfg = vars(config)
    global_step = 0

    # use a resnet18 model
    model = create_model() # student
    ema_model = create_model(ema=True) # teacher

    train_set, val_set, classes = prepare_cifar10(cfg['dataset_root'], cfg['mean'], cfg['std'])
    train_loader, val_loader = sample_train(train_set, val_set, cfg['batch_size'],
                                            cfg['NUM_LABELS'], len(classes), cfg['seed'], cfg['ratio'])

    class_criterion = nn.CrossEntropyLoss(ignore_index=config['NO_LABEL'], reduction='none').cuda()
    consistency_criterion = mse_loss

    optimizer = torch.optim.SGD(model.parameters(), cfg['init_lr'],
                                momentum=0.99, weight_decay=cfg['weight_decay'])

    for epoch in range(cfg['num_epochs']):
        time_meter = AverageMeter()
        batch_time_meter = AverageMeter()

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

            ema_logit = ema_out.detach()

            class_loss = class_criterion(out, target) / minibatch_size
            ema_class_loss = class_criterion(ema_out, target) / minibatch_size

            loss = class_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            update_ema_variable(model, ema_model, cfg['ema_decay'], global_step)

            batch_time_meter.update(time.time() - end_time)
            end_time = time.time()

            if step % 10 == 0:
                print('Epoch: [{0}][{1}]')




