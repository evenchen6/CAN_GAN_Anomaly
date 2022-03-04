import argparse

class Options():
    """
        Param Class
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Run before set
        self.parser.add_argument('--model_name', default='CAN_GAN_Anomaly', help='model name')
        self.parser.add_argument('--abnormal_class', default='Dos', help='Anomaly class str for datasets')
        self.parser.add_argument('--dataset', default='CAN', help='cifar10 | mnist | CAN')
        self.parser.add_argument('--data_root', default='', help='path to dataset')
        self.parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
        self.parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
        self.parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
        self.parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        self.parser.add_argument("--latent_dim", type=int, default=256, help="dimensionality of the latent space")
        self.parser.add_argument("--n_classes", type=int, default=5, help="number of classes for dataset")
        self.parser.add_argument("--n_abnormal_classes", type=int, default=1, help="ODD abnormal num is one, but openset is other")
        self.parser.add_argument("--img_size", type=int, default=48, help="size of each image dimension")
        self.parser.add_argument("--channels", type=int, default=1, help="number of image channels")
        self.parser.add_argument("--is_init_weight", type=bool, default=True, help="is init model weight")
        self.parser.add_argument("--log_path", type=str, default='./log/dataset', help="record log path") # save visualdl log
        self.parser.add_argument("--pkl_path", type=str, default='./pth/dataset', help="record pkl path")
        self.parser.add_argument("--is_train_mode", type=bool, default=False, help="is train mode")

        # Auto Update param
        self.parser.add_argument('--abnormal_class_idx', default=0, help='Anomaly class idx for datasets, auto update!')
        self.parser.add_argument('--normal_class_idx', default=0, help='Normal class idx for datasets, auto update!')
        self.parser.add_argument('--class_to_idx', default='', help='ImageFolder Method class_to_idx, auto update!')
        self.parser.add_argument('--open_dataset_len', default=0, help='open dataset length, auto update!')
        self.parser.add_argument('--close_dataset_len', default=0, help='close dataset length, auto update!')

        self.opt = None

    def parse(self, is_jupyter=False):
        """
            parse param

            Update: support Jupyter environment
        """
        if is_jupyter:
            self.opt = self.parser.parse_known_args()[0]
        else:
            self.opt = self.parser.parse_args()

        return self.opt