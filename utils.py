import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools

import config

def sample_train(train_dataset, val_dataset, batch_size, k, n_classes, seed, label_ratio=0.25):
    '''Randomly form unlabeled data in training dataset

    :param train_dataset:
    :param val_dataset:
    :param batch_size:
    :param k: keep k labeled data in whole training set, other witout label
    :param n_classes:
    :param seed: random seed for shuffle
    :param label_ratio: ratio of labeled samples in a batch, usually a half or a quarter
    :return:
    '''
    n = len(train_dataset) # 60000 for mnist, 50000 for cifar-10
    rrng = np.random.RandomState(seed)
    indices = torch.zeros(k) # indices of labeled data
    others = torch.zeros(n - k) # indices of unlabeled data
    card = k // n_classes
    cpt = 0

    for i in range(n_classes):
        class_items = np.nonzero(np.array(train_dataset.targets) == 0)
        n_class = len(class_items)
        rd = rrng.permutation(np.arange(n_class))
        indices[i * card : (i+1) * card] = torch.squeeze(class_items[rd[:card]])
        others[cpt : cpt+n_class-card] = torch.squeeze(class_items[rd[card:]])
        cpt += (n_class - card)

    # others = others.long()
    train_dataset.targets[others] = config.NO_LABEL

    sampler = TwoStreamSampler(others, indices, batch_size, int(label_ratio*batch_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_sampler=sampler,
                                               num_workers=2,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=config.eval_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)

    return train_loader, eval_loader

class TwoStreamSampler(Sampler):
    """A Sampler used for dataloader,
    It samples two parts samples in a batch which comes from
    different source

    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        '''

        :param primary_indices: unlabeled indices of training set
        :param secondary_indices: labeled indices of training set
        :param batch_size: returned batch size
        :param secondary_batch_size: label samples number in a batch
        '''
        # super(TwoStreamSampler, self).__init__()
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.primary_batch_size = batch_size - secondary_batch_size
        self.secondary_batch_size = secondary_batch_size

        # need to check that given indice list length is enough
        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        """method to iterately return a batch"""
        # majority is unlabeled data
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    '''Collect data into a fixed-length chunks

    :param iterable:
    :param n:
    :return:
    '''
    args = [iter(iterable)] * n

    return zip(*args)

class AverageMeter(object):
    """Compute and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mse_loss(out1, out2):
    assert out1.size() == out2.size()
    quad_diff = torch.sum((F.softmax(out1, dim=1) - F.softmax(out2, dim=1)) ** 2)

    return quad_diff / out1.data.nelement()

def update_ema_variable(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1-(1/(global_step+1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1-alpha)*param.data)