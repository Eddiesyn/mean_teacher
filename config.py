# global vars
dataset_root = '~/datasets/CIFAR-10'
NO_LABEL = -1
NUM_LABELS = 100 # number of labeled images in whole dataset
seed = 10
ratio = 0.25 # ratio of labeled samples dedicated in a batch
ema_decay = 0.999 # exponential moving average decay rate (default 0.999)

# ## for mnist
# mean = (0.1307,)
# std = (0.3081,)
## for cifar10
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

# training vars
batch_size = 64
num_epochs = 50
train_shuffle=True
init_lr = 0.01
weight_decay = 0.0001

# eval vars
eval_size = 16

