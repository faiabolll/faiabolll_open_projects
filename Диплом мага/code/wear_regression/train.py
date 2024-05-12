import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

torch.autograd.set_detect_anomaly(True)

from model import YoloRegression
from dataset import RezcyDataset
from utils import load_checkpoint

from tqdm import tqdm
import time
import os

seed = 123
torch.manual_seed(seed)

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cpu"
BATCH_SIZE = 4
WEIGHT_DECAY = 0
EPOCHS = 7
NUM_WORKERS = 1
PIN_MEMORY = False
LOAD_MODEL = False
LOAD_MODEL_FILE = "model.pt"

IMG_DIR = os.path.join("data", "images")
LABEL_DIR = os.path.join("data", "labels")
ANNOTATIONS_PATH = os.path.join('data', 'test.csv')

    

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
            
        return img
            
transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def reshape_image_meta(data):
    res = list(map(lambda x: x.tolist(), data))
    return torch.tensor(res).T

def train_fn(train_loader, model, optimizer, loss_fn, epoch=None):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x,y) in enumerate(loop):
        img, image_meta = x[0], reshape_image_meta(x[1])
        img, y = img.to(DEVICE), y.to(DEVICE)
        out = model(img, image_meta)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update the progress bar
        loop.set_postfix(loss = loss.item())
        
        if epoch % 5 == 0 and batch_idx == 0:
            # Saving model's state and checkpoints
            torch.save({
                'batch_idx': batch_idx,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                
            }, f'checkpoints\\{LOAD_MODEL_FILE}')
        
    print(f"\nMean loss was {sum(mean_loss) / len(mean_loss)}")
          
def main():
    model = YoloRegression().to(DEVICE)

    if torch.cuda.is_available():
        model.cuda()
          
    optimizer = optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )
          
    loss_fn = torch.nn.MSELoss()
    
    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)
        
    train_dataset = RezcyDataset(
        ANNOTATIONS_PATH,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
        
    test_dataset = RezcyDataset(
        ANNOTATIONS_PATH,
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )
    
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        # shuffle=True,
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
        print(f'Epoch {epoch}')
        
        train_fn(train_loader, model, optimizer, loss_fn, epoch=epoch)
        
if __name__ == '__main__':
    main()