import cv2
from shutil import rmtree
from os.path import join, isfile, basename, exists
from os import walk, makedirs
import re

PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base'
PATH_SAVE = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Model'
width = 320
height = 200
folder_db = 'db_images'

# Delete old folder save
if exists(join(PATH_SAVE, folder_db)):
    print("Deleting old folder: " + join(PATH_SAVE, folder_db))
    rmtree(join(PATH_SAVE, folder_db))

# Resize
print("Resizing ...")
counter = 0
for path, dirs, files in walk(PATH):
    for file in files:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", file):
            try:
                # Open image
                img = cv2.imread(join(path, file), cv2.IMREAD_UNCHANGED)
                path_save = join(PATH_SAVE, folder_db, basename(path) + '_resize')
                # Make directory
                if not exists(path_save):
                    makedirs(path_save)
                # Resize and save image
                cv2.imwrite(join(path_save, file), cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
                counter += 1
                print('\r' + str(counter) + '. ' + file, end='')
            except:
                print("\nCan not resize file: " + file)
