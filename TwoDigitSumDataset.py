import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms

class TwoDigitSumDataset(Dataset):
    def __init__(self, root_dir, train=True, transform=None, dataset_mult = 3, labels=None):
        dataset = torchvision.datasets.MNIST(root_dir, train=train, download=True)

        indexed_ds = {}
        for img, label in dataset:
            if label not in indexed_ds:
                indexed_ds[label] = [img]
            else:
                indexed_ds[label].append(img)
        
        self.transform = transform
        self.dataset_mult = dataset_mult
        self.dataset = []
        self.y_data = []
        self.labels = labels if not train else [indexed_ds[i][0] for i in range(10)]
        self.construct_dataset(dataset, indexed_ds)

    def construct_dataset(self, ds, indexed_val):
        v_data = self.dataset
        y_values = self.y_data
        for _ in range(self.dataset_mult * len(ds)):
            # Get the first number in a randomical way
            idx_x = np.random.randint(0, high=len(indexed_val), size=1)[0]
            value_x = np.random.randint(0, high=len(indexed_val[idx_x]), size=1)[0]
            # Get the second way in a randomical way
            idx_x_1 = np.random.randint(0, high=len(indexed_val), size=1)[0]
            value_x_1 = np.random.randint(0, high=len(indexed_val[idx_x_1]), size=1)[0]
            # Get the two result digits
            first_letter = 1 if (idx_x + idx_x_1) >= 10 else 0
            idx_y = (idx_x + idx_x_1) % len(indexed_val)
            # (first number * second number)
            imgs_enc = (indexed_val[idx_x][value_x], indexed_val[idx_x_1][value_x_1])
            # (first digit * second digit)
            imgs_dec = (self.labels[first_letter], self.labels[idx_y])
            v_data.append((imgs_enc, imgs_dec))
            y_values.append((idx_x, idx_x_1))

    def __len__(self):
        return len(self.dataset)
    
    def get(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_x = self.dataset[idx][0]
        data_y = self.dataset[idx][1]
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # stacking input and output images
        img_end = torch.cat((transform(data_x[0]), transform(data_x[1])), dim=0)
        img_dec = torch.cat((transform(data_y[0]), transform(data_y[1])), dim=0)

        return img_end, img_dec

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_x = self.dataset[idx][0]
        data_y = self.dataset[idx][1]
        # We don't want rotation on output
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # stacking input and output images
        img_end = torch.cat((self.transform(data_x[0]), self.transform(data_x[1])), dim=0)
        img_dec = (self.y_data[idx], torch.cat((transform(data_y[0]), transform(data_y[1])), dim=0))

        return img_end, img_dec