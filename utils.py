import torch
import torch.nn.functional as F
from torch.utils.data.sampler import Sampler
import numpy as np
import itertools
import csv
import PIL.Image as Image
import shutil

import config


def sample_train(train_dataset, val_dataset, n_classes, args):
    """Randomly form unlabeled data in training dataset

    :param train_dataset:
    :param val_dataset:
    :param batch_size:
    :param k: keep k labeled data in whole training set, other witout label
    :param n_classes:
    :param args:
    :return:
    """
    n = len(train_dataset)  # 60000 for mnist, 50000 for cifar-10
    rrng = np.random.RandomState(args.seed)
    indices = torch.zeros(args.NUM_LABELS, dtype=torch.int32)  # indices of labeled data
    others = torch.zeros(n - args.NUM_LABELS, dtype=torch.int32)  # indices of unlabeled data
    card = args.NUM_LABELS // n_classes
    cpt = 0

    for i in range(n_classes):
        class_items = np.flatnonzero(np.array(train_dataset.targets) == i)
        n_class = len(class_items)
        class_items = torch.from_numpy(class_items)
        rd = rrng.permutation(np.arange(n_class))
        indices[i * card: (i + 1) * card] = torch.squeeze(class_items[rd[:card]])
        others[cpt: cpt + n_class - card] = torch.squeeze(class_items[rd[card:]])
        cpt += (n_class - card)

    for i in others:
        train_dataset.targets[i] = args.NO_LABEL

    sampler = TwoStreamSampler(others, indices, args.batch_size, int(args.ratio * args.batch_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_sampler=sampler,
                                               num_workers=2,
                                               pin_memory=True)

    eval_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=args.eval_size,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)

    return train_loader, eval_loader


class RandomTranslateWithReflect:
    """Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TwoStreamSampler(Sampler):
    """A Sampler used for dataloader,
    It samples two parts samples in a batch which comes from
    different source

    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        """

        :param primary_indices: unlabeled indices of training set
        :param secondary_indices: labeled indices of training set
        :param batch_size: returned batch size
        :param secondary_batch_size: label samples number in a batch
        """
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
    """Collect data into a fixed-length chunks

    :param iterable:
    :param n:
    :return:
    """
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


class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in self.header
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.log_file.flush()


def softmax_mse_loss(logits1, logits2):
    assert logits1.size() == logits2.size()
    softmax_1 = F.softmax(logits1, dim=1)
    softmax_2 = F.softmax(logits2, dim=1)

    return F.mse_loss(softmax_1, softmax_2, reduction='sum') / logits1.size()[1]


def softmax_kl_loss(out1, out2):
    assert out1.size() == out2.size()
    input_log_softmax = F.log_softmax(out1, dim=1)
    target_softmax = F.softmax(out2, dim=1)

    return F.kl_div(input_log_softmax, target_softmax, reduction='sum')


def update_ema_variable(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - (1 / (global_step + 1)), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)


def get_current_consistency_weight(epoch, max_weight, rampup_length):
    return max_weight * sigmoid_rampup(epoch, rampup_length)


def sigmoid_rampup(epoch, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        epoch = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - epoch / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def accuracy(output, target, args, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(args.NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / labeled_minibatch_size))
    return res


def save_checkpoint(state, is_best, root, filename):
    torch.save(state, '{}/{}_checkpoint.pth'.format(root, filename))
    if is_best:
        shutil.copyfile('{}/{}_checkpoint.pth'.format(root, filename), '{}/{}_best.pth'.format(root, filename))


def adjust_learning_rate(optimizer, epoch, step, ntrain, args):
    epoch = epoch + step / ntrain
    lr = args.init_lr

    # LR warm-up to handle large minibatch size
    # lr = linear_rampup(epoch, 5) * (init_lr - )

    # Cosine LR rampdown (only one cycle)
    assert args.lr_rampdown_epochs >= args.num_epochs
    lr *= consine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def consine_rampdown(current, rampdown_length):
    assert 0 <= current <= rampdown_length
    return float(0.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value
