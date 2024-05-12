import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as FT
import torch.optim as optim
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

from model import Yolov11
from dataset import RezcyDataset
from loss import Yolov11Loss
from utils import (
    intersection_over_union,
    non_max_suppression,
    mean_average_precision,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint
)

from tqdm import tqdm

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 4
WEIGHT_DECAY = 0
EPOCHS = 2
NUM_WORKERS = 1
PIN_MEMORY = False
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pt"
IMG_DIR = "data\\images\\original"
LABEL_DIR = "data\\labels"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes
            
        return img, bboxes
            
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    
    for batch_idx, (x,y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the progress bar
        loop.set_postfix(loss = loss.item())
        
        # Saving model's state and checkpoints
        # save_checkpoint(model.state_dict(), 'checkpoints\\model\\my_checkpoint.pth.tar')
        # save_checkpoint(optimizer.state_dict(), 'checkpoints\\optimizer\\my_checkpoint.pth.tar')
        torch.save({
            'batch_idx': batch_idx,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
            
        }, f'D:\\yolo3\\checkpoints\\{LOAD_MODEL_FILE}')
        
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")
          
def main():
    model = Yolov11(
        split_size = 7,
        num_boxes = 2
    ).to(DEVICE)
          
    optimizer = optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )
          
    loss_fn = Yolov11Loss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        
    train_dataset = RezcyDataset(
        'example.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=7,
        B=2
    )
        
    test_dataset = RezcyDataset(
        'example.csv',
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
        S=7,
        B=2
    )
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True
    )
    
    for epoch in range(EPOCHS):
        pred_boxes, target_boxes = get_bboxes(
            train_loader,
            model,
            iou_threshold=0.5,
            threshold=0.4
        )
        
        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        
        print(f"Train mAP: {mean_avg_prec}")
        
        train_fn(train_loader, model, optimizer, loss_fn)
        
if __name__ == '__main__':
    main()