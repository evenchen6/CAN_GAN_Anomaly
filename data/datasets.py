"""
SPLIT DATASETS
"""
import torch
import numpy as np

# CIFAR10
def get_cifar_anomaly_dataset(train_ds, valid_ds, close_ds, abn_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - CIFAR10} -- Training dataset
        valid_ds {Dataset - CIFAR10} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        train_ds（include train normal data）
        valid_ds（include test normal + abnormal data）
        close_ds（include test normal data）
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    # process train dataset
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0] # Get normal class index in train data
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    # process test dataset
    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0] # Get normal class index in test data
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    train_ds.data = np.copy(nrm_trn_img)
    valid_ds.data = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)
    close_ds.data = np.copy(nrm_tst_img)

    # close label
    train_ds.targets = np.copy(nrm_trn_lbl)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)
    close_ds.targets = np.copy(nrm_tst_lbl)

    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # open-set label
    train_targets = np.copy(nrm_trn_lbl)
    valid_targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)
    close_targets = np.copy(nrm_tst_lbl)

    return TorchvisonDatasetToOpensetDataset(train_ds, train_targets), TorchvisonDatasetToOpensetDataset(valid_ds, valid_targets), TorchvisonDatasetToOpensetDataset(close_ds, close_targets)

# MNIST
def get_mnist_anomaly_dataset(train_ds, valid_ds, close_ds, abn_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - MNIST} -- Training dataset
        valid_ds {Dataset - MNIST} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        Returns:
        train_ds（include train normal data）
        valid_ds（include test normal/abnormal data + train abnormal data）
        close_ds（include test normal data）
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    # process train dataset
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]  # Get normal class index in train data
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]  # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]  # Abnormal training labels.

    # process test dataset
    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]  # Get normal class index in test data
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]  # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]  # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal training labels.

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    train_ds.data = torch.Tensor(np.copy(nrm_trn_img))
    valid_ds.data = torch.Tensor(np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0))
    close_ds.data = torch.Tensor(np.copy(nrm_tst_img))

    # close label
    train_ds.targets = np.copy(nrm_trn_lbl)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    close_ds.targets = np.copy(nrm_tst_lbl)

    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # open-set label
    train_targets = np.copy(nrm_trn_lbl)
    valid_targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    close_targets = np.copy(nrm_tst_lbl)

    return TorchvisonDatasetToOpensetDataset(train_ds, train_targets), TorchvisonDatasetToOpensetDataset(valid_ds, valid_targets), TorchvisonDatasetToOpensetDataset(close_ds, close_targets)


# CAN
def get_can_anomaly_dataset(train_ds, valid_ds, close_ds, abn_cls_idx=0):
    """"
        CAN Dataset abnormal sample acquisition
        Returns:
            train_ds（include train normal data）
            valid_ds（include test normal + abnormal data）
            close_ds（include test normal data）
    """
    # Get samples
    trn_samples, trn_lbl = np.array(train_ds.samples), np.array(train_ds.targets)
    tst_samples, tst_lbl = np.array(valid_ds.samples), np.array(valid_ds.targets)

    # process train dataset
    nrm_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]  # get train data normal index
    abn_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]  # get train data abnormal index
    nrm_trn_samples = trn_samples[nrm_trn_idx]  # Normal training samples
    abn_trn_sample = trn_samples[abn_trn_idx]  # Abnormal training samples
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]  # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]  # Abnormal training labels.

    # process test dataset
    nrm_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]  # get test data normal index
    abn_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]  # get test data abnormal index
    nrm_tst_samples = tst_samples[nrm_tst_idx]  # Normal training images
    abn_tst_samples = tst_samples[abn_tst_idx]  # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]  # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]  # Abnormal training labels.

    # Assign labels to normal (0) and abnormals (1)
    # Normal as label 0
    # Abnormal as label 1
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal

    # sample feature(data + close label)
    train_ds.samples = np.copy(nrm_trn_samples)
    valid_ds.samples = np.concatenate((nrm_tst_samples, abn_tst_samples), axis=0)
    close_ds.samples = np.copy(nrm_tst_samples)

    # open-set label
    train_ds.targets = np.copy(nrm_trn_lbl)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)
    close_ds.targets = np.copy(nrm_tst_lbl)

    return CANDataset(train_ds), CANDataset(valid_ds), CANDataset(close_ds)

from torch.utils.data import Dataset
class CANDataset(Dataset):
    """
        CAN Encapsulated Dataset(open-set)

        Returns:
            data, normal/abnormal class label, close label
    """

    def __init__(self, dataset):
        self.dataset = dataset  # ImageFolder object, samples include[path,label] targets include normal/abnormal label

    def __getitem__(self, index):
        data, label = self.dataset[index] # exec [], then call __getitem__
        label = label.astype(np.int64)

        return data, self.dataset.targets[index], label

    def __len__(self):
        return len(self.dataset)

class TorchvisonDatasetToOpensetDataset(Dataset):
    """
        torchvision dataset To openset dataset

        Returns:
            data, normal/abnormal class label, close label
    """

    def __init__(self, dataset, targets):
        self.dataset = dataset  # include dataset.data, dataset.targets (data/close label)
        self.targets = targets  # normal/abnormal label

    def __getitem__(self, index):
        data, label = self.dataset[index] # Python exec [], will call __getitem__

        return data, self.targets[index], label

    def __len__(self):
        return len(self.dataset)