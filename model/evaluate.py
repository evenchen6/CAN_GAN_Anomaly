import numpy as np

def Recall(labels, pred):
    """
        Recall

        NOTE: abnormal sample seen as P, so pos_label=1
    :param labels: tensor
    :param pred: tensor
    :return: Recall score
    """
    from sklearn.metrics import recall_score
    labels = labels.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred, axis=1)

    """
        recall_score function input eg.
        >>> y_true = [0, 1, 2, 3]
        >>> y_pred = [0, 2, 1, 3] # Need to convert prediction results into a specific classification
    """
    return recall_score(labels, pred, pos_label=1)


def Precision(labels, pred):
    """
        Precision

        NOTE: abnormal sample seen as P, so pos_label=1
    :param labels: tensor
    :param pred: tensor
    :return: Precision score
    """
    from sklearn.metrics import precision_score
    labels = labels.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred, axis=1)

    """
        precision_score function input eg.
        >>> y_true = [0, 1, 2, 3]
        >>> y_pred = [0, 2, 1, 3] # Need to convert prediction results into a specific classification
    """
    return precision_score(labels, pred, pos_label=1)


def F1Score(labels, pred):
    """
        F1-Score

        NOTE: abnormal sample seen as P, so pos_label=1
    :param labels: tensor
    :param pred: tensor
    :return: F1-Score
    """
    from sklearn.metrics import f1_score
    labels = labels.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred, axis=1)

    """
        f1_score function input eg.
        >>> y_true = [0, 1, 2, 3]
        >>> y_pred = [0, 2, 1, 3] # Need to convert prediction results into a specific classification
    """
    return f1_score(labels, pred, pos_label=1)


def AUC(labels, score):
    """
        AUC is the area under the ROC curve, which is often used for evaluation of openset/ODD image related datasets
        The closer to 1 the better the effect, less than 0.5 theoretically worse than random

    :param labels: tensor
    :param score: tensor
    :return: auc score
    """
    from sklearn.metrics import roc_curve, auc
    labels = labels.cpu().detach().numpy()
    score = score.cpu().detach().numpy()
    # TODO Because the prediction here is the positive/different label [0,1]
    # the label result of the classification can be directly converted into a score, and other models are not applicable
    score = np.argmax(score, axis=1)

    """
        roc_curve function input eg.
        >>> y = np.array([1, 1, 2, 2])
        >>> scores = np.array([0.1, 0.4, 0.35, 0.8])  
    """
    fpr, tpr, _ = roc_curve(labels, score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    return roc_auc


def Accuary(labels, pred):
    """
        Accuary

    :param labels: tensor
    :param pred: tensor
    :return: accuracy
    """
    from sklearn.metrics import accuracy_score
    labels = labels.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    pred = np.argmax(pred, axis=1)

    """
        accuracy_score function input eg.
        >>> y_true = [0, 1, 2, 3]
        >>> y_pred = [0, 2, 1, 3]
    """
    return accuracy_score(labels, pred)


def ResultTransformByStrategy(opt, pre_validity, pre_label, strategy='ACGAN-AC-RFC-Normal', threshold=0.9):
    """
        Fine-grained ODD assumptions only applicable to CAN scenarios

        - The real/fake boundary of D is assumed to be 0.5

        * ACGAN-AC-RFC-Normal: same as GIDS except we have fine-grained classification of attack type
        - Aux1: if maxclass=known attack,  detect known attack and finish; Otherwise (maxclass=normal) pass it to 2nd D
        - D: If fake, Detect unknown attack and finish; Otherwise, declare real=normal

        * ACGAN-AC-RFC-Th: require high-confidence prediction for known attacks and normal
        - Aux1: if largest softmax>th, finish; Otherwise, it is OOD, possibly unknown attack
        - D: If fake, confirm unknown attack; If real,  and aux1 classifies known attack w. low confidence, then conflict with aux1 (use aux1 class result. Should rarely happen). Seems D is useless

        * ACGAN-AC-Th: Verify that the routine operations in the ODD field are all samples whose softmax is less than the threshold are ODD, and those whose softmax is greater than or equal to the current class

    :param opt:
    :param pre_validity: GAN's real/fake label (tensor)
    :param pre_label: Predict label (tensor)
    :param strategy:
    :param threshold:
    :return:
    """
    import torch
    pre_label = torch.exp(pre_label)
    pre_label = pre_label.cpu().detach().numpy()
    pre_validity = pre_validity.cpu().detach().numpy()

    if strategy == "ACGAN-AC-RFC-Normal": # strategy_1
        pred = np.argmax(pre_label, axis=1)  # Take the maximum probability category index of each predicted data
        for index in range(len(pred)):
            if pred[index] == opt.normal_class_idx:
                if pre_validity[index] <= 0.5:
                    pred[index] = opt.abnormal_class_idx

        return pred

    elif strategy == "ACGAN-AC-RFC-Th":
        pred = np.argmax(pre_label, axis=1)  # Take the maximum probability category index of each predicted data
        for index in range(len(pred)):  # Filter the data through the threshold to filter out the non-conforming data
            if pre_label[index, pred[index]] < threshold:
                if pre_validity[index] <= 0.5:
                    pred[index] = opt.abnormal_class_idx

        return pred

    elif strategy == "ACGAN-AC-Th":
        pred = np.argmax(pre_label, axis=1)
        for index in range(len(pred)):  # Filter the data through the threshold to filter out the non-conforming data
            if pre_label[index, pred[index]] < threshold:
                pred[index] = opt.abnormal_class_idx

        return pred

    else:
        # The maximum probability output of the classification network is used by default
        pred = np.argmax(pre_label, axis=1)
        return pred