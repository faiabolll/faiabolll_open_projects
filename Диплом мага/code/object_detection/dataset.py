# DATASET
import torch
import torch.nn as nn
import os
import pandas as pd
from PIL import Image

class RezcyDataset(torch.utils.data.Dataset):
    def __init__(
        self, csv_file, img_dir, label_dir, S=7, B=4, transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.B = B
        self.transform = transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        
        with open(label_path, 'r') as f:
            for label in f.readlines():
                x, y, w, h = [
                    float(i) if float(i) != int(float(i)) else int(i)
                    for i in label.strip().split(' ')
                ]
                boxes.append([x, y, w, h])
                
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)
        
        if self.transform:
            image, boxes = self.transform(image, boxes)
            
            
        label_matrix = torch.zeros((self.S, self.S, 5 * self.B))
        
        for box in boxes:
            x, y, w, h = box
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width, height = self.S * w, self.S * h
            
            if label_matrix[i,j,0] == 0:
                label_matrix[i,j,0] = 1
                label_matrix[i,j,1:5] = torch.tensor([x_cell, y_cell, width, height])
                
        return image, label_matrix