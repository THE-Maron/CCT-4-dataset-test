import torch.utils.data
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from PIL import Image,ImageOps

class EikllxDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, csv_dir, transform=None, channels=1, size=256):
        self.dataset_dir = dataset_dir
        self.transform = transforms.Compose([
            transforms.Resize([size,size]),
            transforms.Grayscale(num_output_channels=channels),
            transforms.ToTensor(),
        ])
        self.dataframe = pd.read_csv(os.path.join(dataset_dir,csv_dir))
        self.labeldict = {'Kai': 0, 'Kan': 1, 'Kin': 2, 'Ten': 3}
        # classes = self.dataframe['label']
        # classes = classes.drop_duplicates()
        # for i in np.arange(len(classes)):
        #     label = classes.iloc[i]
        #     self.labeldict[label] = i

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        element = self.dataframe.iloc[idx]
        img_dir = element['img_dir']
        label = element['label']
        # if your os is windows, Image.open doesn't work due to backslash
        img_dir = img_dir.replace('\\', '/')
        img = Image.open(os.path.join(self.dataset_dir, img_dir)).convert('RGB')
        img = ImageOps.invert(img)

        if self.transform:
            data = self.transform(img)

        return data, self.labeldict[label]

if __name__ == '__main__':
    ds = EikllxDataset('../data','Test.csv')
    print(ds.labeldict)
    print(ds.__len__())
    print(ds.__getitem__(20))