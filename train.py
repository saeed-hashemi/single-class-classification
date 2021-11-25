import torch
from utils.model_tools import Model
from utils.data import get_loaders
from utils.train_tools import train_model


def main(args):
    print('Dataset: {}, Normal Label: {}, LR: {}'.format(
        args.dataset, args.label, args.lr))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = Model(args.backbone)
    model = model.to(device)

    train_loader, test_loader, train_loader_1 = get_loaders(
        dataset=args.dataset, label_class=args.label, batch_size=args.batch_size, backbone=args.backbone)
    train_model(model, train_loader, test_loader, train_loader_1, device, args)


def init_settings():
    class args:
        pass
    args.dataset = 'cifar10'
    args.epochs = 2  # number of epochs
    args.label = '*'  # Use the normal class id or use * for train for each class
    args.lr = 1e-5  # The initial learning rate
    args.batch_size = 32
    args.backbone = 152  # ResNet 18/152
    return args


if __name__ == "__main__":

    # default values:
    args = init_settings()

    # decide to do training just for a class or loop on all of classes
    if args.label == '*':
        labels = list(range(10))
    else:
        labels = [args.label]

    for i in labels:
        main(args)
