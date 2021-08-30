"""
    Model reference ACGAN https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py
    The reference model is suitable for MNIST, which is similar to the CAN dataset, so its network structure is selected as the benchmark
    Our model is an ACGAN model slightly adjusted on the benchmark
"""
import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        # better than one-hot
        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        # Adaptation, want to generate the first linear layer of different output sizes
        if opt.img_size == 32:
            self.init_size = 4
        elif opt.img_size == 48:
            self.init_size = 6
        else:
            raise

        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size * self.init_size))

        """
            Model hierarchy：
            Linear
            (batch, opt.latent_dim(256)) -> (batch, 128 x 6 x 6)

            Upsample layer
            eg1. gen_out_size = 48
            (batch, 128, 6, 6) -> (batch, 64, 12, 12)
            (batch, 64, 12, 12) -> (batch, 32, 24, 24)
            (batch, 32, 24, 24) -> (batch, 16, 48, 48)
            (batch, 16, 48, 48) -[conv]> (batch, opt.channels, 48, 48)
           
            eg2. gen_out_size = 32
            (batch, 128, 4, 4) -> (batch, 64, 8, 8)
            (batch, 64, 8, 8) -> (batch, 32, 16, 16)
            (batch, 32, 16, 16) -> (batch, 16, 32, 32)
            (batch, 16, 32, 32) -[conv]> (batch, opt.channels, 32, 32)
        """
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        """
        Generator forward
        :param noise: random noise
        :param labels: task label(want to generator image label)
        :return: img
        """
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.5)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        """
            Model hierarchy：
            
            Extract feature convolutional layer 
            eg1. input_size = 48
            (batch, opt.channels, 48, 48) -> (batch, 16, 24, 24)
            (batch, 16, 24, 24) -> (batch, 32, 12, 12)
            (batch, 32, 12, 12) -> (batch, 64, 6, 6)
            (batch, 64, 6, 6) -> (batch, 128, 3, 3)
            
            adv_layer: (batch, 128, 3, 3) -[resize]> (batch, 128 x 3 x 3) -> (batch, 1) 
            aux_layer: (batch, 128, 3, 3) -[resize]> (batch, 128 x 3 x 3) -> (batch, 5) 
            
            eg2. input_size = 32
            (batch, opt.channels, 32, 32) -> (batch, 16, 16, 16)
            (batch, 16, 16, 16) -> (batch, 32, 8, 8)
            (batch, 32, 8, 8) -> (batch, 64, 4, 4)
            (batch, 64, 4, 4) -> (batch, 128, 2, 2)
            
            adv_layer: (batch, 128, 2, 2) -[resize]> (batch, 128 x 2 x 2) -> (batch, 1) 
            aux_layer: (batch, 128, 2, 2) -[resize]> (batch, 128 x 2 x 2) -> (batch, 5) 
        """
        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # input_size=48 => ds_size = 3
        # input_size=32 => ds_size = 2
        if opt.img_size == 32:
            ds_size = 2
        elif opt.img_size == 48:
            ds_size = 3
        else:
            raise

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size * ds_size, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size * ds_size, opt.n_classes), nn.LogSoftmax())

    def forward(self, img):
        """
        Discriminator forward
        :param img: input image
        :return: validity, labels
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        labels = self.aux_layer(out)

        return validity, labels
