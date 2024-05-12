from yolov5 import detect
import glob
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torch
import os
import numpy as np
import yaml

# DRIVE_PATH = '/content/drive/MyDrive/yolo'
# last_best = max(os.listdir(os.path.join(DRIVE_PATH, 'models', 'yoloDetect', 'runs', 'train')))

class DetectionModel():
    def __init__(self, model_config, **kwargs):
        self.weights = os.path.join(*model_config['weights_path'])
        self.imgsz = model_config['imgsz']
        self.conf_thres = model_config['conf_thres']
        self.iou_thres = model_config['iou_thres']
        self.source = os.path.join(*model_config['images_folder']) if 'images_folder' not in kwargs.keys() else kwargs['images_folder']
        self.project = os.path.join(*model_config['project_path'])
        self.save_txt = model_config['save_txt']
        self.save_conf = model_config['save_conf']
        
    def run(self, **kwargs):
        detect.run(
            weights=self.weights if 'weights' not in kwargs.keys() else kwargs['weights'], 
            imgsz=self.imgsz if 'imgsz' not in kwargs.keys() else kwargs['imgsz'],
            conf_thres=self.conf_thres if 'conf_thres' not in kwargs.keys() else kwargs['conf_thres'],
            iou_thres=self.iou_thres if 'iou_thres' not in kwargs.keys() else kwargs['iou_thres'],
            source=self.source if 'source' not in kwargs.keys() else kwargs['source'],
            project=self.project if 'project' not in kwargs.keys() else kwargs['project'],
            save_txt=self.save_txt if 'save_txt' not in kwargs.keys() else kwargs['save_txt'],
            save_conf=self.save_conf if 'save_conf' not in kwargs.keys() else kwargs['save_conf']
        )
        

