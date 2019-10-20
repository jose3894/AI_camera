from os import listdir
from os.path import join, isfile
from shutil import copy2

PATH_CHANGE = r'/home/jose8alcaide/Descargas/frame-extraction-master/data/images/video6'
SAVE_PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base/violence'
dest_counter = 1516

for file in listdir(PATH_CHANGE):
        if isfile(join(PATH_CHANGE, file)):
            zero_numbers = 6 - len(str(dest_counter))
            zero_string = '0' * zero_numbers
            print('output' + zero_string + str(dest_counter) + '.png')
            copy2(join(PATH_CHANGE, file), join(SAVE_PATH, 'output' + zero_string + str(dest_counter) + '.png'))
            dest_counter += 1
            print('\routput' + zero_string + str(dest_counter) + '.png')
