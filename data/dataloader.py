"""
LOAD DATA from file.
"""
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, ImageFolder
from data.datasets import get_cifar_anomaly_dataset, get_mnist_anomaly_dataset, get_can_anomaly_dataset
from PIL import Image

class Data:
    """ Dataloader containing train, valid sets and close sets.
    """
    def __init__(self, train, valid, close):
        self.train = train
        self.valid = valid
        self.close = close

##
def load_data(opt):
    """
        Load Data
    """

    ##
    # LOAD DATA SET
    if opt.data_root == '':
        opt.data_root = 'dataset/data'

    def skip_loader(path):
        # load image for pil format(every)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    ## CIFAR
    if opt.dataset in ['cifar10']:
        transform = transforms.Compose([transforms.Resize(opt.img_size),
                                        transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                                        transforms.RandomHorizontalFlip(0.5),
                                        transforms.RandomRotation(10),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                             (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([transforms.Resize(opt.img_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                  (0.2023, 0.1994, 0.2010))])

        train_ds = CIFAR10(root='./dataset', train=True, download=True, transform=transform)
        valid_ds = CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
        close_ds = CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)

        # (data,target,label) => (data，normal/abnormal label，close label)
        opt.abnormal_class_idx = train_ds.class_to_idx[opt.abnormal_class]
        opt.class_to_idx = train_ds.class_to_idx
        train_ds, valid_ds, close_ds = get_cifar_anomaly_dataset(train_ds, valid_ds, close_ds, opt.abnormal_class_idx)

    ## MNIST
    elif opt.dataset in ['mnist']:
        transform = transforms.Compose([transforms.Resize(opt.img_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])


        train_ds = MNIST(root='./dataset', train=True, download=True, transform=transform)
        valid_ds = MNIST(root='./dataset', train=False, download=True, transform=transform)
        close_ds = MNIST(root='./dataset', train=False, download=True, transform=transform)

        # (data,target,label) => (data，normal/abnormal label，close label)
        opt.abnormal_class_idx = int(opt.abnormal_class)
        opt.class_to_idx = train_ds.class_to_idx
        train_ds, valid_ds, close_ds = get_mnist_anomaly_dataset(train_ds, valid_ds, close_ds, opt.abnormal_class_idx)

    # CAN dataset
    elif opt.dataset in ['CAN']:
        transform = transforms.Compose([transforms.Resize(opt.img_size),
                                        transforms.ToTensor()])

        # use transform to tensor
        train_ds = ImageFolder(root='dataset/data/train', transform=transform, loader=skip_loader)
        valid_ds = ImageFolder(root='dataset/data/test', transform=transform, loader=skip_loader)
        close_ds = ImageFolder(root='dataset/data/test', transform=transform, loader=skip_loader)

        print(train_ds.class_to_idx)

        # (data,target,label) => (data，normal/abnormal label，close label)
        # NOTE: this code only can dataset valid(other dataset need to adapt)
        opt.abnormal_class_idx = train_ds.class_to_idx[opt.abnormal_class]
        opt.normal_class_idx = train_ds.class_to_idx["Normal"]
        opt.class_to_idx = train_ds.class_to_idx
        train_ds, valid_ds, close_ds = get_can_anomaly_dataset(train_ds, valid_ds, close_ds, opt.abnormal_class_idx)

    else:
        pass

    # Auto Update
    opt.open_dataset_len = len(valid_ds)
    opt.close_dataset_len = len(close_ds)

    train_dl = DataLoader(dataset=train_ds, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    close_dl = DataLoader(dataset=close_ds, batch_size=opt.batch_size, shuffle=False, drop_last=False)
    return Data(train_dl, valid_dl, close_dl)