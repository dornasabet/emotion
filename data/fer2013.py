import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from data.dataset import CustomDataset, MultiTaskDataset
from PIL import Image


def load_data(path='datasets/fer2013/fer2013.csv'):
    fer2013 = pd.read_csv(path)
    affectnet_expr2emotion = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust',
                              6: 'Anger', 7: 'Contempt'}
    idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                    7: 'Surprise'}
    class_to_idx = {cls: idx for idx, cls in idx_to_class.items()}
    return fer2013, affectnet_expr2emotion


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


affectnet_expr2emotion = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust',
                          6: 'Anger', 7: 'Contempt'}
idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                7: 'Surprise'}
# idx_to_class={0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}
class_to_idx = {cls: idx for idx, cls in idx_to_class.items()}


def get_dataloaders(path_train='train_d.csv', path_val='val_d.csv', train_dir="./data/train",
                    val_dir="./data/val", bs=64, augment=True):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation

        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    # fer2013_train, emotion_mapping = load_data(path_train)
    # fer2013_val, emotion_mapping = load_data(path_train)
    #
    # xtrain, ytrain = prepare_data(fer2013[fer2013['Usage'] == 'Training'])
    # xval, yval = prepare_data(fer2013[fer2013['Usage'] == 'PrivateTest'])

    mu, st = 0, 255

    test_transform = transforms.Compose([
        # transforms.Scale(52),
        transforms.TenCrop(220),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(
            lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),

            transforms.TenCrop(220),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(
                lambda tensors: torch.stack([transforms.Normalize(mean=(mu,), std=(st,))(t) for t in tensors])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
        ])
    else:
        train_transform = test_transform

    # X = np.vstack((xtrain, xval))
    # Y = np.hstack((ytrain, yval))

    train = MultiTaskDataset(csv_file=path_train, root=train_dir, transform=train_transform)
    val = MultiTaskDataset(csv_file=path_val, root=val_dir, transform=test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=64, shuffle=False, num_workers=0)

    return trainloader, valloader
