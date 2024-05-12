import os, glob
from PIL import Image

DRIVE_PATH = '/content/drive/MyDrive/yolo'

image_paths = glob(os.path.join(DRIVE_PATH, 'original','*.png'))
label_paths = [path.replace('original', 'labels').replace('.png', '.txt') for path in image_paths]

box_image_path = os.path.join(DRIVE_PATH, 'regImages')
box_label_path = os.path.join(DRIVE_PATH, 'regLabels')

def read_image_from_path(path):
    image = Image.open(path)
    return image

def read_label_from_path(path):
    with open(path, 'r') as f:
        content = f.readlines() # ["0 0.1 0.1 0.1 0.1 0.5\n", "0 0.2 0.2 0.2 0.2 0.5\n"]
    return content

def extract_boxes(image, labels):
    im_width, im_height = image.size
    boxes = []
    for label in labels:
        x,y,w,h,wr = label.strip().split(' ')
        left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2
        left, top, right, bottom = int(im_width * left), int(im_height * top), int(im_width * right), int(im_height * bottom)
        box = image.crop((left, top, right, bottom))
        box = [box, x,y,w,h, wr]
        boxes.append(box)
    return boxes

def upload_box(box, image_path, label_path, num_box):
    image_ix = os.path.basename(image_path).split('.')[0]
    box_image_name = num_box + '_' + image_ix + '.png'
    box_label_name = num_box + '_' + image_ix + '.txt'
    image, label = box[0], box[1:]

    image.save(os.path.join(image_path, box_image_name))

    with open(os.path.join(label_path, box_label_name), 'r') as f:
        f.write(' '.join(label))


for image_path, label_path in zip(image_paths, label_paths):
    image_obj = read_image_from_path(image_path)
    label_table = read_label_from_path(label_path)
    boxes = extract_boxes(image_obj, label_table)
    for num_box, box in enumerate(boxes): # box = [crop_image, x,y,w,h, wr]
        upload_box(box, box_image_path, box_label_path, num_box)
