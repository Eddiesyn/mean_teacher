import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    # begin for video application
    parser.add_argument('--labeled_video_path', default='/data1/UCF-101_img', type=str, help='vids path of labeled dataset(ucf101)')
    parser.add_argument('--labeled_annotation_path', default='/usr/home/sut/datasets/UCF/ucf101_01.json',
                        type=str, help='annotation path of labeled dataset')
    parser.add_argument('--unlabeled_video_path', default='/data1/kinetics-600', type=str, help='vids path of unlabeled dataset(kinetics)')
    parser.add_argument('--unlabeled_annotation_path', default='/usr/home/sut/Efficient-3DCNNs/annotation_Kinetics/kinetics.json',
                        type=str, help='annotation path of unlabeled dataset')
    parser.add_argument('--initial_scale', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--scale_step', default=0.84089641525, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    # end for video application
    parser.add_argument('--dataset_root', default='', type=str, help='Root directory path of data')
    parser.add_argument('--result_path', default='./results/', type=str, help='result path')
    parser.add_argument('--pth_name', default='cifar10_resnet18', type=str, help='chief name')
    parser.add_argument('--NO_LABEL', default=-1, type=int, help='ignore value')
    parser.add_argument('--NUM_LABELS', default=4000, type=int, help='total number of labeled data')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--ratio', default=0.25, type=float, help='ratio of labeled samples dedicated in each batch')
    parser.add_argument('--ema_decay', default=0.99, type=float, help='ema decay rate')
    parser.add_argument('--consistency_type', default='mse', type=str, help='either mse or kl')
    parser.add_argument('--consistency_weight', default=100, type=int, help='maximum value of ramupup consistency weight')
    parser.add_argument('--rampup_length', default=5, type=int, help='number of epochs of rampup')
    parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=300, type=int, help='total number of training epochs')
    parser.add_argument('--init_lr', default=0.2, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=2.0e-4, type=float, help='l2 weight decay')
    parser.add_argument('--eval_size', default=32, type=int, help='evaluation batch size')
    parser.add_argument('--resume_path', default='', type=str, help='path of .pth for resume training')
    parser.add_argument('--lr_rampdown_epochs', default=350, type=int, help='length of learning rate cosine rampdown (>= length of training)')

    args = parser.parse_args()

    return args