import cv2
from os.path import join


INPUT_VIDEO_PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Model/videos/video_accidents.mp4'
OUTPUT_IMAGES_PATH = r'/home/jose8alcaide/Documentos/AI_camera/AI_app/Data_Base/test_images/Accidents'
counter_image = 0
counter_frame = 0

# initialize the video stream, pointer to output video file, and
# frame dimensions
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