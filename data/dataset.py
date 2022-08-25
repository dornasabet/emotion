import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])

        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


affectnet_expr2emotion = {0: 'Neutral', 1: 'Happiness', 2: 'Sadness', 3: 'Surprise', 4: 'Fear', 5: 'Disgust',
                          6: 'Anger', 7: 'Contempt'}
idx_to_class = {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happiness', 5: 'Neutral', 6: 'Sadness',
                7: 'Surprise'}
class_to_idx = {cls: idx for idx, cls in idx_to_class.items()}


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root, transform):
        df = pd.read_csv(csv_file)
        df = df[df['category'].isin(affectnet_expr2emotion.keys())]
        self.paths = [os.path.basename(fp) for fp in df.image]
        self.targets = np.array([class_to_idx[affectnet_expr2emotion[expr]] for expr in df.category])
        self.valence_arousal = df[['valnce', 'arousal']].to_numpy()
        self.transform = transform
        self.root = root

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        # dealing with the labels
        emotion_label = self.targets[idx]
        valence = torch.tensor(float(self.valence_arousal[idx, 0]), dtype=torch.float32)
        arousal = torch.tensor(float(self.valence_arousal[idx, 1]), dtype=torch.float32)

        # dealing with the image
        img = Image.open(os.path.join(self.root, idx_to_class[emotion_label], self.paths[idx])).convert('RGB')
        img = self.transform(img)
        # print(self.paths[idx])
        # print(torch.tensor(float(self.valence_arousal[idx,0]), dtype=torch.float32))
        # print(torch.tensor(float(self.valence_arousal[idx,1]), dtype=torch.float32))

        return img.data, (emotion_label, valence, arousal)
