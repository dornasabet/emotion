import warnings

import torch
from torch.cuda.amp import autocast
from utils.loss import total_loss, RMSE, SAGR, CCC, PCC

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(net, dataloader, optimizer, scaler, Ncrop=True):
    net = net.train()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    iters = len(dataloader)  # number of batches, not images
    running_rmse_valence = running_sagr_valence = running_pcc_valence = running_ccc_valence = 0.0
    running_rmse_arousal = running_sagr_arousal = running_pcc_arousal = running_ccc_arousal = 0.0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.to(device)
        labels[0] = labels[0].type(torch.LongTensor)
        labels = [l.to(device) for l in labels]
        with autocast():
            if Ncrop:
                # fuse crops and batchsize
                bs, ncrops, c, h, w = inputs.shape
                inputs = inputs.view(-1, c, h, w)

            # repeat labels ncrops times
            labels[0] = torch.repeat_interleave(labels[0], repeats=ncrops, dim=0)
            labels[1] = torch.repeat_interleave(labels[1], repeats=ncrops, dim=0)
            labels[2] = torch.repeat_interleave(labels[2], repeats=ncrops, dim=0)

            # forward + backward + optimize
            outputs = net(inputs)
            p_class = outputs[:, :8]
            p_valence = outputs[:, 8]
            p_arousal = outputs[:, 8 + 1]
            OUT = [p_class, p_valence, p_arousal]
            loss = total_loss(OUT, labels).float()
            # loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # scheduler.step(epoch + i / iters)

            # calculate performance metrics
            loss_tr += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs.data, 1)
            correct_count += (preds == labels).sum().item()
            rmse_valense = RMSE(p_valence, labels[1])
            running_rmse_valence += rmse_valense * inputs.size(0)
            sagr_valence = SAGR(p_valence, labels[1])
            running_sagr_valence += sagr_valence * inputs.size(0)
            pcc_valence = PCC(p_valence, labels[1])
            running_pcc_valence += pcc_valence * inputs.size(0)
            ccc_valence = CCC(p_valence, labels[1])
            running_ccc_valence += ccc_valence * inputs.size(0)
            rmse_arousal = RMSE(p_arousal, labels[2])
            running_rmse_arousal += rmse_arousal * inputs.size(0)
            sagr_arousal = SAGR(p_arousal, labels[2])
            running_sagr_arousal += sagr_arousal * inputs.size(0)
            pcc_arousal = PCC(p_arousal, labels[2])
            running_pcc_arousal += pcc_arousal * inputs.size(0)
            ccc_arousal = CCC(p_arousal, labels[2])
            running_ccc_arousal += ccc_arousal * inputs.size(0)
            n_samples += inputs.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples
    epoch_vrmse = running_rmse_valence / n_samples
    epoch_vsagr = running_sagr_valence / n_samples
    epoch_vpcc = running_pcc_valence / n_samples
    epoch_vccc = running_ccc_valence / n_samples
    epoch_armse = running_rmse_arousal / n_samples
    epoch_asagr = running_sagr_arousal / n_samples
    epoch_apcc = running_pcc_arousal / n_samples
    epoch_accc = running_ccc_arousal / n_samples

    return acc, loss, epoch_vrmse, epoch_vsagr, epoch_vpcc, epoch_vccc, epoch_armse, epoch_asagr, epoch_apcc, epoch_accc


def evaluate(net, dataloader, Ncrop=True):
    net = net.eval()
    loss_tr, correct_count, n_samples = 0.0, 0.0, 0.0
    running_rmse_valence = running_sagr_valence = running_pcc_valence = running_ccc_valence = 0.0
    running_rmse_arousal = running_sagr_arousal = running_pcc_arousal = running_ccc_arousal = 0.0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        if Ncrop:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            # forward

            outputs = net(inputs)
            p_class = outputs[:, :8]
            p_valence = outputs[:, 8]
            p_arousal = outputs[:, 8 + 1]
            # combine results across the crops
            p_class = p_class.view(bs, ncrops, -1)
            p_valence = p_valence.view(bs, ncrops, -1)
            p_arousal = p_arousal.view(bs, ncrops, -1)

            p_class = torch.sum(p_class, dim=1) / ncrops
            p_valence = torch.sum(p_valence, dim=1) / ncrops
            p_arousal = torch.sum(p_arousal, dim=1) / ncrops

            OUT = [p_class, p_valence, p_arousal]

            # outputs = torch.sum(outputs, dim=1) / ncrops
        else:
            outputs = net(inputs)
            p_class = outputs[:, :8]
            p_valence = outputs[:, 8]
            p_arousal = outputs[:, 8 + 1]
            OUT = [p_class, p_valence, p_arousal]

        loss = total_loss(OUT, labels).float()
        # loss = criterion(outputs, labels)

        # calculate performance metrics
        loss_tr += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs.data, 1)
        correct_count += (preds == labels).sum().item()
        rmse_valense = RMSE(p_valence, labels[1])
        running_rmse_valence += rmse_valense * inputs.size(0)
        sagr_valence = SAGR(p_valence, labels[1])
        running_sagr_valence += sagr_valence * inputs.size(0)
        pcc_valence = PCC(p_valence, labels[1])
        running_pcc_valence += pcc_valence * inputs.size(0)
        ccc_valence = CCC(p_valence, labels[1])
        running_ccc_valence += ccc_valence * inputs.size(0)
        rmse_arousal = RMSE(p_arousal, labels[2])
        running_rmse_arousal += rmse_arousal * inputs.size(0)
        sagr_arousal = SAGR(p_arousal, labels[2])
        running_sagr_arousal += sagr_arousal * inputs.size(0)
        pcc_arousal = PCC(p_arousal, labels[2])
        running_pcc_arousal += pcc_arousal * inputs.size(0)
        ccc_arousal = CCC(p_arousal, labels[2])
        running_ccc_arousal += ccc_arousal * inputs.size(0)
        n_samples += inputs.size(0)

    acc = 100 * correct_count / n_samples
    loss = loss_tr / n_samples
    epoch_vrmse = running_rmse_valence / n_samples
    epoch_vsagr = running_sagr_valence / n_samples
    epoch_vpcc = running_pcc_valence / n_samples
    epoch_vccc = running_ccc_valence / n_samples
    epoch_armse = running_rmse_arousal / n_samples
    epoch_asagr = running_sagr_arousal / n_samples
    epoch_apcc = running_pcc_arousal / n_samples
    epoch_accc = running_ccc_arousal / n_samples

    return acc, loss, epoch_vrmse, epoch_vsagr, epoch_vpcc, epoch_vccc, epoch_armse, epoch_asagr, epoch_apcc, epoch_accc
