import sys
import warnings

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.fer2013 import get_dataloaders
from utils.checkpoint import save
from utils.hparams import setup_hparams
from utils.loops import train, evaluate
from utils.setup_network import setup_network
from utils.loss import total_loss

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(net, logger, hps):
    # Create dataloaders
    trainloader, valloader = get_dataloaders(bs=hps['bs'])

    net = net.to(device)

    learning_rate = float(hps['lr'])
    scaler = GradScaler()

    # optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.0001, amsgrad=True)
    # optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.0001)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(trainloader), epochs=hps['n_epochs'])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6, last_epoch=-1, verbose=False)
    # criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    print("Training", hps['name'], "on", device)
    for epoch in range(hps['start_epoch'], hps['n_epochs']):

        acc_tr, loss_tr, vrmse_tr, vsagr_tr, vpcc_tr, vccc_tr, armse_tr, asagr_tr, apcc_tr, accc_tr = train(
            net, trainloader, optimizer, scaler)
        logger.loss_train.append(loss_tr)
        logger.acc_train.append(acc_tr)
        logger.vrmse_train.append(vrmse_tr)
        logger.vsagr_train.append(vsagr_tr)
        logger.vpcc_train.append(vpcc_tr)
        logger.vccc_train.append(vccc_tr)
        logger.armse_train.append(armse_tr)
        logger.asagr_train.append(asagr_tr)
        logger.apcc_train.append(apcc_tr)
        logger.accc_train.append(accc_tr)

        acc_v, loss_v, vrmse_v, vsagr_v, vpcc_v, vccc_v, armse_v, asagr_v, apcc_v, accc_v = evaluate(
            net, valloader)
        logger.loss_val.append(loss_v)
        logger.acc_val.append(acc_v)
        logger.vrmse_val.append(vrmse_v)
        logger.vsagr_val.append(vsagr_v)
        logger.vpcc_val.append(vpcc_v)
        logger.vccc_val.append(vccc_v)
        logger.armse_val.append(armse_v)
        logger.asagr_val.append(asagr_v)
        logger.apcc_val.append(apcc_v)
        logger.accc_val.append(accc_v)

        # Update learning rate
        scheduler.step(acc_v)

        if acc_v > best_acc:
            best_acc = acc_v

            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        if (epoch + 1) % hps['save_freq'] == 0:
            save(net, logger, hps, epoch + 1)
            logger.save_plt(hps)

        print('Epoch %2d' % (epoch + 1),
              'Train Accuracy: %2.4f %%' % acc_tr,
              'Val Accuracy: %2.4f %%' % acc_v,
              sep='\t\t')

    # Calculate performance on test set
    acc_test, loss_test = evaluate(net, testloader, criterion)
    print('Test Accuracy: %2.4f %%' % acc_test,
          'Test Loss: %2.6f' % loss_test,
          sep='\t\t')


if __name__ == "__main__":
    # Important parameters
    hps = setup_hparams(sys.argv[1:])
    logger, net = setup_network(hps)

    run(net, logger, hps)
