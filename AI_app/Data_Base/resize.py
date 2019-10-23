import cv2
from os import listdir
from os.path import join, isfile

PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base/river'
PATH_SAVE = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base/river_resize'
width = 28
height = 21

counter = 0
for file in listdir(PATH):
        if isfile(join(PATH, file)):
            try:
                img = cv2.imread(join(PATH, file), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(join(PATH_SAVE, file), cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))
                counter += 1
                print('\r' + str(counter) + '. ' + file, end='')
            except:
                print("\nCan not resize file: " + file)
