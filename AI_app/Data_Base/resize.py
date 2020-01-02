import cv2
from shutil import rmtree
from os.path import join, isfile, basename, exists
from os import walk, makedirs
import re
import numpy as np
import random

PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base/google_images/guns'
PATH_SAVE = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Model'
width = 224
height = 224
folder_db = 'db_images3'

# Delete old folder save
if exists(join(PATH_SAVE, folder_db)):
    print("Deleting old folder: " + join(PATH_SAVE, folder_db))
    rmtree(join(PATH_SAVE, folder_db))

# Resize
print("Making new images ...")
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

                # Split name
                name_file = file.split('.png')[0]

                # Resize and save image
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(join(path_save, name_file + '_Rsz_.png'), img)

                counter += 1

                # Translate and save image
                """
                tx = random.randint(0, 100)
                ty = random.randint(0, 100)
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                translate_img = cv2.warpAffine(img, M, (320, 200))
                cv2.imwrite(join(path_save, name_file + '_T_.png'), translate_img)

                counter += 1
                """

                # Rotate and save image
                rows, cols = img.shape[:2]
                scale = random.uniform(0.8, 2)
                #angle = random.randint(20, 360)
                angle = random.randint(1, 3)
                angle = angle * 90
                M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
                rotate_img = cv2.warpAffine(img, M, (cols, rows))
                cv2.imwrite(join(path_save, name_file + '_Rot_.png'), rotate_img)

                counter += 1

                # Filter Gauss
                g_img = cv2.GaussianBlur(img, (7, 7), 2000)
                cv2.imwrite(join(path_save, name_file + '_G_.png'), g_img)

                counter += 1

                # Noise
                """
                gauss = np.random.normal(0, 1, img.size)
                gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
                n_img = cv2.add(img, gauss)
                cv2.imwrite(join(path_save, name_file + '_N_.png'), n_img)
                
                counter += 1
                """
                print('\r' + str(counter) + '. ' + file, end='')
            except:
                print("\nCan not make new images: " + file)
