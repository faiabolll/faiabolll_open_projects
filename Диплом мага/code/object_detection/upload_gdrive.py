from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # client_secrets.json need to be in the same directory as the script
drive = GoogleDrive(gauth)

def upload_images_to_google_drive(
    image_num, 
    mask_colored_folder_name='mask colored',
    mask_folder_name='mask',
    original_folder_name='original',
):
    folderName = 'yolo'  # Please set the folder name.

    folders = drive.ListFile(
        {'q': "title='" + folderName + "' and mimeType='application/vnd.google-apps.folder' and trashed=false"}).GetList()
    for folder in folders:
        if folder['title'] == folderName:
            subfolders = drive.ListFile({'q': f"'{folder['id']}' in parents and trashed=false"}).GetList()
            for subf in subfolders:
                if subf['title'] == mask_colored_folder_name:
                    mask_colored_folder = subf
                    continue
                if subf['title'] == mask_folder_name:
                    mask_folder = subf
                    continue
                if subf['title'] == original_folder_name:
                    original_folder = subf
                    continue
    
    file_png = drive.CreateFile({
        'parents': [{
            'id': original_folder['id'], 
            'kind': 'drive#fileLink'
        }], 'mimeType': 'image/x-png', 
        'title': f'{image_num}.png'
    })
    file_png.SetContentFile(f'data\\images\\{original_folder_name}\\{image_num}.png')
    file_png.Upload()
    
    file_png = drive.CreateFile({
        'parents': [{
            'id': mask_folder['id'], 
            'kind': 'drive#fileLink'
        }], 'mimeType': 'image/x-png', 
        'title': f'{image_num}.png'
    })
    file_png.SetContentFile(f'data\\images\\{mask_folder_name}\\{image_num}.png')
    file_png.Upload()
    
    file_exr = drive.CreateFile({
        'parents': [{
            'id': mask_colored_folder['id'], 
            'kind': 'drive#fileLink'
        }], 'mimeType': 'image/x-exr', 
        'title': f'{image_num}.exr'
    })
    file_exr.SetContentFile(f'data\\images\\{mask_colored_folder_name}\\{image_num}.exr')
    file_exr.Upload()
    
if __name__ == '__main__':
    # upload_images_to_google_drive('361244')
    print('main')