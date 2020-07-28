from torch.utils.data import Dataset,DataLoader
import os
from PIL import Image
import numpy as np
import torch

class MyDataset(Dataset):
    mean = [0.6712, 0.5770, 0.5549]
    std = [0.2835, 0.2785, 0.2641]
    def __init__(self,path):
        self.path = path
        self.dataset = os.listdir(path)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        name = self.dataset[index]
        img = Image.open(os.path.join(self.path, name))
        img = np.array(img) / 255.
        img = (img - MyDataset.mean) / MyDataset.std
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        return img

if __name__ == '__main__':

    imagelist = os.listdir(r"F:\迅雷下载\11月班\2020-05_15_GAN\faces")

    data = MyDataset(r"F:\迅雷下载\11月班\2020-05_15_GAN\faces")
    loader = DataLoader(dataset=data,batch_size=51223,shuffle=True)
    data= next(iter(loader))
    mean = torch.mean(data, dim=(0,2,3))
    std = torch.std(data, dim=(0,2,3))
    print(mean ,std)