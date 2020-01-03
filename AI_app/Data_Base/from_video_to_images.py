import cv2
from os import makedirs
from shutil import rmtree
from os.path import join, exists


INPUT_VIDEO_PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Model/videos/video_violence.mp4'
OUTPUT_IMAGES_PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base/test_images/violence'
counter_image = 0
counter_frame = 0

# Delete old folder and make new one
if exists(OUTPUT_IMAGES_PATH):
    print("Deleting old folder: " + OUTPUT_IMAGES_PATH)
    rmtree(OUTPUT_IMAGES_PATH)
makedirs(OUTPUT_IMAGES_PATH)

print("[INFO] processing video...")
vs = cv2.VideoCapture(INPUT_VIDEO_PATH)
writer = None

# loop over frames from the video file stream
while True:
    # read the next frame from the file
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    counter_frame += 1

    # Save image
    if counter_frame > 10:
        counter_image += 1
        name_file = 'output_' + str(counter_image) + '.png'
        cv2.imwrite(join(OUTPUT_IMAGES_PATH, name_file), frame)
        counter_frame = 0


# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()