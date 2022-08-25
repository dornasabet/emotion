import numpy as np
import torch
import torch.nn as nn

import random


def ACC(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    # ground_truth= np.squeeze(ground_truth.cpu().numpy())
    # predictions= np.squeeze(predictions.cpu().numpy())
    return torch.mean(ground_truth.astype(int) == predictions.astype(int))


def RMSE(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    # ground_truth= np.squeeze(ground_truth.cpu().numpy())
    # predictions= np.squeeze(predictions.cpu().numpy())
    return torch.sqrt(torch.mean((ground_truth - predictions) ** 2))


def SAGR(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    # ground_truth= np.squeeze(ground_truth.cpu().numpy())
    # predictions= np.squeeze(predictions.cpu().numpy())
    # print(np.sign(ground_truth.detach().clone().cpu().numpy()) == np.sign(predictions.detach().clone().cpu().numpy()))
    a = torch.sign(ground_truth) == torch.sign(predictions)
    a = a.float()
    # print("-----------------")
    # print(np.mean(np.sign(ground_truth.detach().clone().cpu().numpy()) == np.sign(predictions.detach().clone().cpu().numpy())))
    # print(torch.mean(a))
    return torch.mean(a)


def PCC(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    ground_truth_ = np.squeeze(ground_truth.clone().detach().cpu().numpy())
    predictions_ = np.squeeze(predictions.clone().detach().cpu().numpy())
    # print(ground_truth_.shape, predictions_.shape)
    # print(np.corrcoef(ground_truth_, predictions_))
    # print(torch.stack((ground_truth, predictions),0).shape)

    # print(torch.corrcoef(torch.stack((ground_truth, predictions),0))[0,1])
    return torch.corrcoef(torch.stack((ground_truth, predictions), 0))[0, 1]


def CCC(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    # ground_truth_= np.squeeze(ground_truth.cpu().numpy())
    # predictions_= np.squeeze(predictions.cpu().numpy())
    mean_pred = torch.mean(predictions)
    mean_gt = torch.mean(ground_truth)

    std_pred = torch.std(predictions)
    std_gt = torch.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0 * pearson * std_pred * std_gt / (std_pred ** 2 + std_gt ** 2 + (mean_pred - mean_gt) ** 2)


def total_loss(ground_truth, predictions):
    loss_cat = nn.CrossEntropyLoss()
    alpha = np.random.uniform(low=0.0, high=1.0, size=None)
    beta = np.random.uniform(low=0.0, high=1.0, size=None)
    gama = np.random.uniform(low=0.0, high=1.0, size=None)
    total = alpha + beta + gama

    catrgorical = loss_cat(ground_truth[0], predictions[0])
    mse = RMSE(ground_truth[1], predictions[1]) + RMSE(ground_truth[2], predictions[2])
    pcc = 1 - ((PCC(ground_truth[1], predictions[1]) + PCC(ground_truth[2], predictions[2])) / 2)
    ccc = 1 - ((CCC(ground_truth[1], predictions[1]) + CCC(ground_truth[2], predictions[2])) / 2)
    loss = catrgorical + (alpha / total) * mse + (beta / total) * pcc + (gama / total) * ccc
    return loss

# def total_loss(ground_truth, predictions):
#     loss_cat=nn.CrossEntropyLoss()
#     catrgorical=loss_cat(ground_truth[0], predictions[0])
#     print("cat")
#     mse=RMSE(np.squeeze(ground_truth[1].cpu().numpy()), np.squeeze(predictions[1].cpu().numpy()))+RMSE(np.squeeze(ground_truth[2].cpu().numpy()), np.squeeze(predictions[2].cpu().numpy()))
#     print("mse")
#     pcc= 1-((PCC(np.squeeze(ground_truth[1].cpu().numpy()), np.squeeze(predictions[1].cpu().numpy()))+PCC(np.squeeze(ground_truth[2].cpu().numpy()), np.squeeze(predictions[2].cpu().numpy())))/2)
#     print(pcc)
#     ccc= 1-((CCC(np.squeeze(ground_truth[1].cpu().numpy()), np.squeeze(predictions[1].cpu().numpy()))+CCC(np.squeeze(ground_truth[2].cpu().numpy()), np.squeeze(predictions[2].cpu().numpy())))/2)
#     print(ccc)
#     loss= catrgorical+ (1/3)*(mse+ pcc+ ccc)
#     return loss
