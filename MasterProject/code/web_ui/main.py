# git clone https://github.com/ultralytics/yolov5
# pip install -U -r yolov5/requirements.txt --quiet

# main app modules
from flask import Flask, render_template, request
import pandas as pd
from PIL import Image, ImageDraw

import numpy as np
import cv2

# Google Drive authorization
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# utils
import yaml, os, shutil, glob, json
from drive_utils import *
from modelDetect import DetectionModel
from modelRegression import YoloRegression

import torch
import torchvision.transforms as transforms

# /**************** CONSTANTS AND FILTERS INITIALIZATION ****************/
# AVALIABLE IN DRIVE
# with open(os.path.join('data', 'available_images.csv'), 'r') as f:
#     available_images = list(map(lambda name: int(name.strip()), f.readlines()))
# AVAILABLE LOCALLY

DISPLAY_IMAGES_NUM = 10
available_images = [
    int(name.replace('.png', '')) 
    for name in os.listdir(os.path.join('data', 'images'))
    ]


configs = pd.read_csv('configs.csv')
# GET FILTERS IF SUCH IMAGES EXISTS
configs = configs[configs.index.isin(available_images)]

for col in configs.columns:
    if col in ['rotation_angle', 'wear_ratio', 'unwear_height']:
        configs[col] = configs[col].astype('float')
    else:
        configs[col] = configs[col].astype('str')


# FILTERS = {}
# for col in configs.columns:
#     if configs[col].dtype != 'float':
#         FILTERS[col] = {'data':list(configs[col].unique()), 'dtype':'str'}
#     else:
#         FILTERS[col] = {'data':sorted(list(configs[col].unique())), 'dtype':'float'}
FILTERS = {}
res = []
for row in configs.iterrows():
    res.append(str(row[0]))
FILTERS = {"image_index": res}


with open('models_config.yaml', 'r') as f:
    MODELS_CONFIG = yaml.safe_load(f)



# /**************** FLASK ROUTING ****************/
app = Flask(__name__)

@app.route('/ui')
def launch_UI():
    if len(request.args) == 0:
        return get_base_template()
    else:
        return get_template_with_images(request.args)
    
def get_base_template():
    data = {}
    return render_template(
        'index.html',
        filters=FILTERS,
        data=data 
    )

def get_template_with_images(params):
    data = get_data_for_render(params)
    return render_template(
        'index.html',
        filters=FILTERS,
        data=data # data: [images_with_annotations, annotations]
    )

def get_data_for_render(params):
    image_name = str(params['image_index']).zfill(7)+'.png'

    # RUN DETECTION (saves multiple images)
    paths_to_image, labels = run_detect(image_name=image_name)

    detection_results = []
    for im, lab in zip(paths_to_image, labels):
        detection_results.append([im, lab])
    # RUN REGRESSION
    regression_results = run_regression(detection_results) # {'1.png': ['c,x,y,w,h,conf,wr], '2.png': ['c,x,y,w,h,conf,wr]}

    drawed_imgs_annotations = dict()
    for reg_res in regression_results.items():
        # print("3333333333333333333333333", reg_res[0])
        config = configs.loc[int(reg_res[0].split('.')[0]),:].to_dict()
        config['image_name'] = reg_res[0]
        drawed_imgs_annotations[reg_res[0]] = (create_image_to_render(reg_res), config)

    return {'response': detection_results, 'drawed_imgs_annotations':drawed_imgs_annotations} # [image_to_draw, annotations, config_data]


def create_image_to_render(image_data):
    image_filename, annotations = image_data
    img = cv2.imread(os.path.join(*MODELS_CONFIG['source_images_folder'], image_filename), cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.putText(img, image_filename, (0,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,bottomLeftOrigin=False)
    im_width, im_height, _ = img.shape

    models_results = []
    for i, annotation in enumerate(annotations):
        _,x,y,w,h,conf,wr = list(map(float, annotation.split(' ')))
        left = int(im_width * (x - w / 2))
        bottom = int(im_height * (y + h / 2))
        right = int(im_width * (x + w / 2))
        top = int(im_height * (y - h / 2))
        
        cv2.rectangle(img, (left, top), (right, bottom), (255,0,0), 2)
        cv2.putText(img, str(i), (left, top), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,bottomLeftOrigin=False)
        models_results.append((i, conf, wr))

    cv2.imwrite(os.path.join('static', 'images', 'analysed', image_filename), img)
    
    return models_results

# /**************** UI BUILD ****************/
def get_images_to_draw(image_num: int):
    """Takes 10 neighbour images to display"""
    configs_copy = configs.copy().reset_index()
    configs_ix = configs_copy[configs_copy['index'] == image_num].index[0]
    if DISPLAY_IMAGES_NUM + configs_ix <= configs_copy.shape[0] - 1:
        start,end = configs_ix, configs_ix+DISPLAY_IMAGES_NUM
    else:
        start,end = configs_ix - (DISPLAY_IMAGES_NUM - (configs_copy.shape[0] - configs_ix)), configs_copy.shape[0]
    return configs.iloc[start:end, :]

# /**************** INITIALIZATIONS ****************/

def initialize_drive():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    return drive

# DETECTION MODEL    
with open('models_config.yaml', 'r') as f:
    model_config = yaml.safe_load(f)

modelDetect = DetectionModel(
    model_config['detection'], 
    images_folder=os.path.join(*model_config['detection']['images_folder'])
)

# REGRESSION MODEL
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
            
        return img
            
class RegressionModel():
    def __init__(self) -> None:
        self.model = YoloRegression()
        self.transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

    def initialize(self, get_last_run=True, get_last_epoch=True):
        if get_last_run:
            folder = sorted(glob.glob(os.path.join('regression', 'train', '*')), key=lambda x: x.split('_')[0], reverse=True)[0]
            checkpoints = glob.glob(os.path.join(folder, 'weights', '*.pt'))
        if get_last_epoch:
            weights = sorted(
                checkpoints, 
                key=lambda x: int(os.path.basename(x.replace('epoch', '').replace('.pt', ''))),
                reverse=True
                )[0]
        checkpoint = torch.load(weights)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def _reshape_image_meta(self, data):
        # if type(data) != 'list':
        #     res = list(map(lambda x: x.tolist() if type(x) != 'list' else x, data))
        res = torch.tensor(data)
        # print(res)
        return res
    
    def __call__(self, img, image_meta):
        if self.transform:
            img = self.transform(img)
        # print("SSSSSSSSSSS", img.shape, "FFFFFFFFFF", len(image_meta))
        img, image_meta = img[None,...], self._reshape_image_meta(image_meta)[None,...]
        with torch.no_grad():
            out = self.model(img, image_meta)
            return out
            
modelRegression = RegressionModel()
modelRegression.initialize()

# /**************** DATA CREATION ****************/

def clear_detect_exps():
    for folder in os.listdir(os.path.join(*model_config['detection']['project_path'])):
        shutil.rmtree(os.path.join(*model_config['detection']['project_path'], folder))

def copy_file(src, dest):
    try:
        shutil.copy(src, dest)
    except shutil.SameFileError:
        pass

def run_detect(image_name, **kwargs):
    copy_file(
        os.path.join(*model_config['detection']['source_images_folder'], image_name),
        os.path.join(*model_config['detection']['images_folder'], image_name)
    )
    # clear previous detection runs result
    clear_detect_exps()
    modelDetect.run()
    paths_to_image, labels = [], []
    # MOVE IMAGE TO STATIC
    for file in glob.glob(os.path.join(*model_config['detection']['project_path'], 'exp', '*.png')):
        copy_file(
            file,
            os.path.join('static', 'images', os.path.basename(file))
        )
        paths_to_image.append(os.path.join('static', 'images', os.path.basename(file)))
    # MOVE LABELS TO STATIC
    for file in glob.glob(os.path.join(*model_config['detection']['project_path'], 'exp', 'labels', '*.txt')):
        copy_file(
            file,
            os.path.join('static', 'labels', os.path.basename(file))
        )

    for path in paths_to_image:
        label_filename = os.path.basename(path).replace('.png', '.txt')
        with open(os.path.join('static', 'labels', label_filename), 'r') as f:
            content = list(map(lambda x: x.strip(), f.readlines()))
        labels.append(content)
    return (paths_to_image, labels)

def run_regression(detects, **kwargs):
    regression_results = {}
    for img_path, labels in detects:
        src_img = os.path.join(
            *MODELS_CONFIG['regression']['source_images_folder'],
            os.path.basename(img_path)
            )
        img = Image.open(src_img)
        im_width, im_height = img.size
        boxes_wr = []
        for label in labels:
            c,x,y,w,h,conf = list(map(float, label.strip().split(' ')))
            box = img.crop((
                int(im_width * (x - w / 2)),
                int(im_height * (y - h / 2)),
                int(im_width * (x + w / 2)),
                int(im_height * (y + h / 2))
            ))
            wr = modelRegression(box, [x,y,w,h])
            boxes_wr.append(' '.join(list(map(str, [c,x,y,w,h,conf,wr.item()]))))

        regression_results[os.path.basename(img_path)] = boxes_wr
        
    return regression_results


if __name__ == '__main__':
    # google drive authorization
    # drive = initialize_drive()

    # flask initialized and launched
    app.run(host='127.0.0.1', port=5000)
