import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of GAN models.")

    parser.add_argument('--model', type=str, default='DCGAN', choices=['DCGAN', 'WGAN-GP'])
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', default='/mnt/dataset1')
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--download', type=str, default='False')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')

    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=78100, help='The number of iterations for generator in WGAN model.')
    parser.add_argument('--nm', action='store_true')
    parser.add_argument('--sat', action='store_true')
    return check_args(parser.parse_args())

# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    args.channels = 3

    return args
