import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import TransformTwice, RandomTranslateWithReflect


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
