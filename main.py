from config import parse_args
from data_loader import get_data_loader

from dcgan import DCGAN_MODEL
from wgan_gradient_penalty import WGAN_GP


def main(args):
    model = None
    if args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-GP':
        model = model = WGAN_GP(args)

    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)