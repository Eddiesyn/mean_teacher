import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import numpy as np

from utils import TransformTwice, RandomTranslateWithReflect, TwoStreamSampler
from datasets import ucf101, kinetics


def prepare_mnist(root, transform):
    # Don't use drop last and shuffle
    train_dataset = datasets.MNIST(root=root, train=True, transform=transform, download=True)
    val_dataset = datasets.MNIST(root=root, train=False, transform=transform, download=True)

    return train_dataset, val_dataset


def prepare_cifar10(root):
    transform_train = TransformTwice(transforms.Compose([
        RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]))
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_val)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_dataset, val_dataset, classes



class UCF_with_Kinetics(Dataset):
    def __init__(self,
                 labeled_root_path,
                 labeled_annotation_path,
                 unlabeled_root_path,
                 unlabeled_annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 labeled_vids_loader=ucf101.get_default_video_loader,
                 unlabeled_vids_loader=kinetics.get_default_video_loader):
        self.data, self.labeled_class_names = ucf101.make_dataset(
            labeled_root_path, labeled_annotation_path, subset, n_samples_for_each_video, sample_duration
        )
        self.unlabeled_data, _ = kinetics.make_dataset(unlabeled_root_path,
                                                       unlabeled_annotation_path,
                                                       subset, n_samples_for_each_video, sample_duration)
        self.labeled_length = len(self.data)
        self.unlabeled_length = len(self.unlabeled_data)
        # concatenate two data list
        self.data.extend(self.unlabeled_data)

        self.label_loader = labeled_vids_loader()
        self.unlabeled_loader = unlabeled_vids_loader()

        self.spatial_transform=spatial_transform
        self.temporal_transform=temporal_transform
        self.target_transform=target_transform

        self.sample_duration = sample_duration

    def __getitem__(self, index):
        path = self.data[index]['video']
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        if index < self.labeled_length: # is ucf vids data
            clip = self.label_loader(path, frame_indices)
            target = self.data[index]['label']
        else: # is kinetics data
            clip = self.unlabeled_loader(path, frame_indices, self.sample_duration)
            target = -1

        if self.spatial_transform is not None:
            self.spatial_transform.ramdomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1,0,2,3)

        return clip, target

    def __len__(self):
        return len(self.data)


def prepare_ucf_and_kinetics(opt, spatial_trainsform, temporal_transform, target_transform):
    sp_transform = TransformTwice(spatial_trainsform)
    temp_transform = TransformTwice(temporal_transform)

    training_data = UCF_with_Kinetics(labeled_root_path=opt.labeled_video_path,
                                      labeled_annotation_path=opt.labeled_annotation_path,
                                      unlabeled_root_path=opt.unlabeled_video_path,
                                      unlabeled_annotation_path=opt.unlabeled_annotation_path,
                                      subset='training', spatial_transform=sp_transform,
                                      temporal_transform=temp_transform,
                                      target_transform=target_transform)
    label_length = training_data.labeled_length
    unlabel_length = training_data.unlabeled_length
    assert label_length + unlabel_length == len(training_data), 'Fatal error in Dataset'

    return training_data, label_length, unlabel_length


def generate_combined_loader(opt, train_dataset, val_dataset, label_length):
    label_indices = torch.from_numpy(np.arange(label_length))
    unlabel_indices = torch.from_numpy(np.arange(label_length, len(train_dataset)))

    sampler = TwoStreamSampler(label_indices, unlabel_indices,
                               opt.batch_size, int(opt.ratio * opt.batch_size))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_sampler=sampler,
                                               num_workers=2, pin_memory=True)
    eval_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=16,
                                              shuffle=False,
                                              num_workers=2, pin_memory=True)

    return train_loader, eval_loader