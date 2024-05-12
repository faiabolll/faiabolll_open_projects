# Google Drive authorization
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

from io import BytesIO
from PIL import Image
import yaml
import os
import re


def download_train_data(drive, folderId):
    exp_folders = drive.ListFile(
        {'q': f"'{folderId}' in parents and trashed=false"}
    ).GetList() 

    for exp_folder in exp_folders:
        exp_folderTitle, exp_folderId = exp_folder['title'], exp_folder['id']
        local_exp_folder = os.path.join('detect', 'train', exp_folderTitle)
        local_weights_folder = os.path.join(local_exp_folder, 'weights')

        # CHEKPOINTS PATH
        if not os.path.exists(local_exp_folder):
            os.mkdir(local_exp_folder)

        # WEIGHTS PATH
        if not os.path.exists(local_weights_folder):
            os.mkdir(local_weights_folder)
        
        exp_files =  drive.ListFile(
            {'q': f"'{exp_folderId}' in parents and trashed=false"}
        ).GetList()

        # DOWNLOAD CHECKOPOINTS AND WEIGHTS
        for exp_file in exp_files:
            exp_file_title = exp_file['title']
            # CHECKPOINTS
            if re.search(r'events\.out\.tfevents|\w{3}\.yaml', exp_file_title):
                exp_file.GetContentFile(os.path.join(local_exp_folder, exp_file_title))
            # WEIGHTS
            if exp_file_title == 'weights':
                pt_files = drive.ListFile(
                    {'q': f"'{exp_file['id']}' in parents and trashed=false"}
                ).GetList()

                for pt_file in pt_files:
                    pt_file.GetContentFile(os.path.join(local_weights_folder, pt_file['title']))

def download_regression_data(drive, folderId):
    exp_folders = drive.ListFile(
        {'q': f"'{folderId}' in parents and trashed=false"}
    ).GetList() 

    for exp_folder in exp_folders:
        exp_folderTitle, exp_folderId = exp_folder['title'], exp_folder['id']
        local_exp_folder = os.path.join('regression', 'train', exp_folderTitle.replace(':','='))
        local_weights_folder = os.path.join(local_exp_folder, 'weights')

        # CHEKPOINTS PATH
        if not os.path.exists(local_exp_folder):
            os.mkdir(local_exp_folder)

        # WEIGHTS PATH
        if not os.path.exists(local_weights_folder):
            os.mkdir(local_weights_folder)
        
        exp_files =  drive.ListFile(
            {'q': f"'{exp_folderId}' in parents and trashed=false"}
        ).GetList()

        # DOWNLOAD CHECKOPOINTS AND WEIGHTS
        for exp_file in exp_files:
            
            exp_file_title = exp_file['title']
            print(exp_file_title)
            # CHECKPOINTS
            if re.search(r'events\.out\.tfevents|\w{3}\.yaml', exp_file_title):
                exp_file.GetContentFile(os.path.join(local_exp_folder, exp_file_title))
            # WEIGHTS
            if exp_file_title == 'weights':
                pt_files = drive.ListFile(
                    {'q': f"'{exp_file['id']}' in parents and trashed=false"}
                ).GetList()

                for pt_file in pt_files:
                    pt_file.GetContentFile(os.path.join(local_weights_folder, pt_file['title']))


def download_avaliable_images(drive, folderId):
    image_names = drive.ListFile(
        {'q': f"'{folderId}' in parents and trashed=false"}
    ).GetList() 

    available_images = '\n'.join([image_name['title'].replace('.png', '') for image_name in image_names])
    with open(os.path.join('data', 'available_images.csv'), 'w') as f:
        f.write(available_images)

def download_image(drive, folderId, image_num):
    if type(image_num) == 'int':
        image_num = str(image_num).zfill(7)+'.png'
    elif type(image_num) == 'str' and '.png' in image_num:
        assert len(image_num.split('.')[0]) == 7

    image_names = drive.ListFile(
        {'q': f"'{folderId}' in parents and trashed=false"}
    ).GetList() 

    for image in image_names:
        if image['title'] == image_num:
            image.GetContentFile(os.path.join('temp', 'drive.img'))
            break
    
    img = Image.open(BytesIO(image.content.getvalue()))
    return img

def download_images(drive, folderId, image_nums:list):
    image_names = drive.ListFile(
        {'q': f"'{folderId}' in parents and trashed=false"}
    ).GetList() 
    if type(image_nums[0]) == int:
        image_nums = list(map(lambda x: str(x).zfill(7)+'.png',image_nums))
    elif type(image_nums[0]) == str:
        assert len(image_nums[0].split('.')[0]) == 7

    res = []
    for image in image_names:
        if image['title'] in image_nums:
            image.GetContentFile(os.path.join('temp', 'drive.img'))
            res.append(Image.open(BytesIO(image.content.getvalue())))

    return res
    

if __name__ == '__main__':
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
    drive = GoogleDrive(gauth)
    with open('models_config.yaml', 'r') as f:
        models_config = yaml.safe_load(f)
    # download_train_data(drive, models_config['detection']['runs/train_folder_id'])
    # download_avaliable_images(drive, models_config['drive_image_source_id'])
    # res = download_images(drive, models_config['drive_image_source_id'], [542354, 1524836])
    download_regression_data(drive, models_config['regression']['runs/train_folder_id'])