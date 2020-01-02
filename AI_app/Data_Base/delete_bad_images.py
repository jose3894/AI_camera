from os import walk, remove
from os.path import join
import cv2

PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Model/db_images/violence/'
counter = 0

for path, dirs, files in walk(PATH):
        for file in files:
            if cv2.imread(join(path, file), cv2.IMREAD_UNCHANGED) is None:
                counter += 1
                remove(join(path, file))
        print(str(counter) + ' images deletes in folder ' + path)
        counter = 0


