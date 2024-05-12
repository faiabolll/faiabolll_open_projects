# DATASET
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from PIL import Image


class RezcyDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, transform=None
    ):
        self.annotations = pd.read_csv(csv_file, sep=';')
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        with open(label_path, 'r') as f:
            params = f.readlines()[-1].strip().split(' ')
            params = list(map(float, params))
            x, y, w, h, wr = params
            
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        return (image, (x,y,w,h)), wr
        


# DEBUG
import torchvision.transforms as transforms
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
            
        return img
            
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])


if __name__ == '__main__':
    print('started debug')
    ds = RezcyDataset('data\\test.csv', 'data\\images', 'data\\labels', transform=transform)
    dds = DataLoader(ds)
    for t in dds:
        print(t)

    # def __getitem__(self, index):
    #     label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
    #     boxes = []
        
    #     with open(label_path, 'r') as f:
    #         for label in f.readlines():
    #             x, y, w, h, wr = [
    #                 float(i) if float(i) != int(float(i)) or ix == 4 else int(i)
    #                 for ix, i in enumerate(label.strip().split(' '))
    #             ]
    #             boxes.append([x, y, w, h, wr])
                
        
    #     img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
    #     image = Image.open(img_path)
    #     boxes = torch.tensor(boxes)
        
    #     if self.transform:
    #         image, boxes = self.transform(image, boxes)
            
            
    #     label_matrix = torch.zeros((self.S, self.S, 6 * self.B))
        
    #     for box in boxes:
    #         x, y, w, h, wr = box
    #         i, j = int(self.S * y), int(self.S * x)
    #         x_cell, y_cell = self.S * x - j, self.S * y - i
    #         width, height = self.S * w, self.S * h
            
    #         if label_matrix[i,j,0] == 0:
    #             label_matrix[i,j,0] = 1
    #             label_matrix[i,j,1:5] = torch.tensor([x_cell, y_cell, width, height])
    #             label_matrix[i,j,6] = wr
                
    #     return image, label_matrix