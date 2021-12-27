import time
import torch
from networks import Generator, Discriminator
from random import randomint_plus
from evaluate import ResultTransformByStrategy
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from visualdl import LogWriter
import os

def weights_init_normal(model):
    """
        NOTE: Currently only support initialize Conv/BatchNormal
    """
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)

class Model:
    """
        Model class, initialize G/D + loss + weight
    """
    def __init__(self, opt, dataloader):
        self.opt = opt
        self.dataloader = dataloader
        self.device = torch.device("cpu")

        # Define model
        self.generator = Generator(opt)
        self.discriminator = Discriminator(opt)

        # Defining the loss function Adversarial loss/Auxiliary classification loss/Positive anomaly classification loss
        self.adversarial_loss = torch.nn.BCELoss()
        self.auxiliary_loss = torch.nn.NLLLoss()

        # If there is cuda, convert the model parameters + loss function (not necessary) to cuda memory
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.device = torch.device("cuda:0")

            self.generator = self.generator.cuda(self.device)
            self.discriminator = self.discriminator.cuda(self.device)
            self.adversarial_loss = self.adversarial_loss.cuda(self.device)
            self.auxiliary_loss = self.auxiliary_loss.cuda(self.device)

        # Define Tensor
        self.FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

        # Initialize model parameters
        if opt.is_init_weight:
            self.generator.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)

        # Construct model parameter optimizer
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(), lr=0.001, momentum=0.9)
        self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones=[200], gamma=0.2) # Learning rate decay

    def train(self):
        """
            Overall training process
        :return:
        """
        print(f">> Training {self.opt.model_name} on {self.opt.dataset} to detect {self.opt.abnormal_class}")
        if self.opt.is_train_mode:
            print(f">> Mode is Train")
        else:
            print(f">> Mode is Val")

        now_time = str(time.time())  # Used to distinguish each training data log + model

        # When visualdl displays, it defaults to the log with the latest timestamp in the same folder as the record
        # So it needs to be written into one folder uniformly
        self.writer_metrics = LogWriter(logdir=os.path.join(self.opt.log_path, now_time, "metrics"))
        self.writer_images = LogWriter(logdir=os.path.join(self.opt.log_path, now_time, "images"))
        self.writer_loss = LogWriter(logdir=os.path.join(self.opt.log_path, now_time, "loss"))

        # Train n_epochs times
        for epoch in range(self.opt.n_epochs):

            # Train once here / evaluate once
            if self.opt.is_train_mode: # Train mode
                self.train_one_epoch(epoch)
            else: # Test mode
                self.load_pkl(epoch)

            self.val_openset(epoch)

            print("[Epoch %d/%d]" % (epoch, self.opt.n_epochs))

            # Save process picture
            self.gen_sample(epoch)

            # In training mode, save the model for each epoch
            if self.opt.is_train_mode:
                os.makedirs(os.path.join(self.opt.pkl_path, now_time), mode=0o777, exist_ok=True)
                torch.save(self.discriminator, os.path.join(self.opt.pkl_path, now_time, ("discriminator_epoch_%d.pkl" % epoch)))
                torch.save(self.generator, os.path.join(self.opt.pkl_path, now_time, ("generator_epoch_%d.pkl" % epoch)))

        self.writer_metrics.close()
        self.writer_images.close()
        self.writer_loss.close()

    def load_pkl(self, epoch = 0):
        """
            Load the trained model
        Args:
            epoch:

        Returns:

        """
        # Construct model parameter path
        discriminator_pkl_path = os.path.join(self.opt.pkl_path, ("discriminator_epoch_%d.pkl" % epoch))
        generator_pkl_path = os.path.join(self.opt.pkl_path, ("generator_epoch_%d.pkl" % epoch))

        # Load model parameters
        discriminator = torch.load(discriminator_pkl_path, map_location=self.device)
        generator = torch.load(generator_pkl_path, map_location=self.device)

        # If there is cuda, switch to cuda
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            discriminator = discriminator.cuda(self.device)
            generator = generator.cuda(self.device)

        self.discriminator = discriminator
        self.generator = generator

    def train_one_epoch(self, epoch = 0):
        """
            Single epoch training process

            NOTE: Do not transmit data, save copy time
        :return:
        """
        self.discriminator.train()
        self.generator.train()

        for i, dataloader in enumerate(tqdm(self.dataloader.train, leave=False, total=len(self.dataloader.train))):
            # Original data training feature/positive anomaly classification label/true closed set label
            # At the same time, convert to cpu/cuda according to the situation
            origin_imgs, _, origin_label = self.set_input(dataloader)

            # Expected confrontation prediction result (true image: 1 generated image: 0)
            adversarial_real = Variable(self.FloatTensor(self.opt.batch_size, 1).fill_(1.0), requires_grad=False)
            adversarial_fake = Variable(self.FloatTensor(self.opt.batch_size, 1).fill_(0.0), requires_grad=False)

            """
                Train G to generate samples of known types       
            """
            self.optimizer_G.zero_grad()

            # Randomly sample z from the normal distribution and randomly generate known class labels
            # (remove abnormal class index)
            z = Variable(self.FloatTensor(np.random.normal(0, 1, (self.opt.batch_size, self.opt.latent_dim))))
            gen_labels = Variable(self.LongTensor(randomint_plus(low=0, high=self.opt.n_classes, cutoff=self.opt.abnormal_class_idx, size=(self.opt.batch_size, ))))

            # Generate data with known labels
            gen_imgs = self.generator(z, gen_labels)

            # Predicted loss + loss record
            pre_validity, pre_labels = self.discriminator(gen_imgs)
            g_RFC = self.adversarial_loss(pre_validity, adversarial_real)
            g_AC = self.auxiliary_loss(pre_labels, gen_labels)
            g_loss = 0.5 * (g_AC + g_RFC)
            self.writer_loss.add_scalar(tag="Loss/g_loss", step=(epoch * len(self.dataloader.train) + i), value=g_loss.cpu())
            self.writer_loss.add_scalar(tag="Loss/g_RFC", step=(epoch * len(self.dataloader.train) + i), value=g_RFC.cpu())
            self.writer_loss.add_scalar(tag="Loss/g_AC", step=(epoch * len(self.dataloader.train) + i), value=g_AC.cpu())

            # BP + update model parameters
            g_loss.backward()
            self.optimizer_G.step()

            """
                Train D to optimize prediction
            """
            self.optimizer_D.zero_grad()

            # Real data prediction
            pre_validity, pre_labels = self.discriminator(origin_imgs)
            d_origin_RFC = self.adversarial_loss(pre_validity, adversarial_real)
            d_origin_AC = self.auxiliary_loss(pre_labels, origin_label)
            d_origin_loss = (d_origin_RFC + d_origin_AC) / 2

            # Generate-known category data prediction
            pre_validity, pre_labels = self.discriminator(gen_imgs.detach())
            d_gen_RFC = self.adversarial_loss(pre_validity, adversarial_fake)
            d_gen_AC = self.auxiliary_loss(pre_labels, gen_labels)
            d_gen_loss = (d_gen_RFC + d_gen_AC) / 2

            # Loss summary + loss record
            d_loss = (d_origin_loss + d_gen_loss) / 2
            self.writer_loss.add_scalar(tag="Loss/d_loss", step=(epoch * len(self.dataloader.train) + i), value=d_loss.cpu())
            self.writer_loss.add_scalar(tag="Loss/d_RFC", step=(epoch * len(self.dataloader.train) + i), value=((d_origin_RFC + d_gen_RFC) / 2).cpu())
            self.writer_loss.add_scalar(tag="Loss/d_AC", step=(epoch * len(self.dataloader.train) + i), value=((d_origin_AC + d_gen_AC) / 2).cpu())

            # BP + update model parameters
            d_loss.backward()
            self.optimizer_D.step()

        # Fix the use of attenuation, attenuate with epoch granularity
        self.scheduler_D.step()


    def val_openset(self, epoch = 0):
        """
            Model evaluation
            - Macro: Recall / Precision / f1-score / Accuracy
            - Self: Recall / Precision / f1-score

        :return: recall / precision / f1_score / auc / accuracy
        """
        self.discriminator.eval()
        self.generator.eval()

        """
            Mixed data set (open set) evaluation, here mainly use macro index evaluation
        """
        recall, precision, f1_score, accuracy = 0, 0, 0, 0

        # Used to record the prediction results of the entire validation data set
        origin_label_record = torch.zeros(size=(self.opt.open_dataset_len, ), dtype=torch.float, device=self.device)
        pre_label_record = torch.zeros(size=(self.opt.open_dataset_len, self.opt.n_classes), dtype=torch.float, device=self.device)
        pre_validity_record = torch.zeros(size=(self.opt.open_dataset_len, 1), dtype=torch.float, device=self.device)

        for i, data in enumerate(self.dataloader.valid, start=0):
            # Load open set data and convert
            img, _, label = self.set_input(data)
            pre_validity, pre_label = self.discriminator(img)

            # Record the true label and predicted label of each row of data in the current batch
            origin_label_record[i * self.opt.batch_size: i * self.opt.batch_size + label.size(0)] = label
            pre_label_record[i * self.opt.batch_size: i * self.opt.batch_size + pre_label.size(0)] = pre_label
            pre_validity_record[i * self.opt.batch_size: i * self.opt.batch_size + pre_validity.size(0)] = pre_validity

        # Obtain the final classification results according to different new class inspection strategies
        # (multiple model combinations + some threshold strategies)
        origin_label_record = origin_label_record.cpu().detach().numpy()  # Prevent abnormal calculation indicators in GPU scenarios
        strategy_options = ['ACGAN-AC-RFC-Normal', 'ACGAN-AC-RFC-Th', 'ACGAN-AC-Th']
        threshold_options = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
        for strategy_option in strategy_options:
            if strategy_option == "ACGAN-AC-RFC-Normal":
                tag_name = strategy_option
                transform_pre_label_record = ResultTransformByStrategy(self.opt, pre_validity_record, pre_label_record, strategy_option)
                self.val_visualdl_record(origin_label_record, transform_pre_label_record, epoch, tag_name)
            else:
                for threshold_option in threshold_options:
                    tag_name = strategy_option + "_threshold_" + str(threshold_option)
                    transform_pre_label_record = ResultTransformByStrategy(self.opt, pre_validity_record, pre_label_record, strategy_option, threshold_option)
                    self.val_visualdl_record(origin_label_record, transform_pre_label_record, epoch, tag_name)


    def val_visualdl_record(self, origin_label_record, transform_pre_label_record, epoch=0, tag=''):
        """
            Record visual data during the verification process (extract from val_openset, adapt to multiple strategies)

        :param origin_label_record:
        :param transform_pre_label_record:
        :param epoch:
        :param tag:
        :return:
        """
        # The memory space of the copy used to prevent downstream use to modify upstream data, resulting in inaccurate experiments
        origin_label_record = origin_label_record.copy()
        transform_pre_label_record = transform_pre_label_record.copy()

        # Overall result evaluation index (scikit-learn/macro)
        from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
        recall = recall_score(origin_label_record, transform_pre_label_record, average='macro')
        precision = precision_score(origin_label_record, transform_pre_label_record, average='macro')
        f1 = f1_score(origin_label_record, transform_pre_label_record, average='macro')
        accuracy = accuracy_score(origin_label_record, transform_pre_label_record)
        # Record and visualize training historical indicators
        self.writer_metrics.add_scalar(tag=tag + "/Mix/Recall", step=epoch, value=recall)
        self.writer_metrics.add_scalar(tag=tag + "/Mix/Precision", step=epoch, value=precision)
        self.writer_metrics.add_scalar(tag=tag + "/Mix/F1Score", step=epoch, value=f1)
        self.writer_metrics.add_scalar(tag=tag + "/Mix/Accuracy", step=epoch, value=accuracy)

        # Single category evaluation indicator (single) + record visualization
        for index in range(self.opt.n_classes):
            # Data preprocessing
            # scikit-learn's two-category indicators only support scenarios where the data label is two-category
            origin_label_single_record = origin_label_record.copy()
            origin_label_single_idx = np.where(origin_label_single_record != index)[0]
            origin_label_single_record[origin_label_single_idx] = self.opt.n_classes

            transform_pre_label_single_record = transform_pre_label_record.copy()
            transform_pre_label_single_idx = np.where(transform_pre_label_single_record != index)[0]
            transform_pre_label_single_record[transform_pre_label_single_idx] = self.opt.n_classes

            recall_single = recall_score(origin_label_single_record, transform_pre_label_single_record, pos_label=index)
            precision_single = precision_score(origin_label_single_record, transform_pre_label_single_record, pos_label=index)
            f1_score_single = f1_score(origin_label_single_record, transform_pre_label_single_record, pos_label=index)
            self.writer_metrics.add_scalar(tag=tag + "/" + list(self.opt.class_to_idx)[index] + "/Recall", step=epoch,
                                           value=recall_single)
            self.writer_metrics.add_scalar(tag=tag + "/" + list(self.opt.class_to_idx)[index] + "/Precision", step=epoch,
                                           value=precision_single)
            self.writer_metrics.add_scalar(tag=tag + "/" + list(self.opt.class_to_idx)[index] + "/F1Score", step=epoch,
                                           value=f1_score_single)

    def set_input(self, data):
        """
            Convert the data to the memory/video memory space where the device is located
        :param data: Minimum batch data
        :return:
        """
        img, targets, label = data
        return img.to(self.device), targets.to(self.device), label.to(self.device)

    def write_file(self, file_name, data, mode = 'w'):
        """
            Write data to file, model can be specified
        :param file_name:
        :param data:
        :return:
        """
        if os.path.exists(file_name):
            f = open(file_name, mode=mode)
        else:
            f = open(file_name, mode='w')
        f.write(data)
        f.close()

    def gen_sample(self, epoch = 0):
        """
            Generate each type of image to check the accuracy of the generation
        Returns:
            gen_known_imgs[np.ndarray]
        """
        self.generator.eval()

        # Generate a known image
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (self.opt.n_classes - self.opt.n_abnormal_classes, self.opt.latent_dim))))
        known_label = list(range(self.opt.n_classes))
        known_label.remove(self.opt.abnormal_class_idx)
        known_labels = Variable(self.LongTensor(known_label))
        gen_known_imgs = self.generator(z, known_labels)  # => torch.float32
        gen_known_imgs = np.array([img.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy() for img in gen_known_imgs])

        # Use visualdl to record and visualize training historical indicators
        # Because there is a problem with the batch increase of single-channel data
        # it is modified here to single data record
        gen_known_imgs_count = 0
        for index in range(self.opt.n_classes):
            if index != self.opt.abnormal_class_idx:
                self.writer_images.add_image(tag='Gen/' + list(self.opt.class_to_idx)[index], step=epoch, img=gen_known_imgs[gen_known_imgs_count])
                gen_known_imgs_count += 1

        return gen_known_imgs