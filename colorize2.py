# Import statements
import numpy as np
import cv2
import os
import time 

# Paths to load the model
DIR = r"F:\DTU\Sem 6\IT420 Computer Vision\Colorize"
PROTOTXT = os.path.join(DIR, r"model/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"model/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"model/colorization_release_v2.caffemodel")

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def FrameCapture(path):
  
    # Path to video file
    vidObj = cv2.VideoCapture(path)
  
    # Used as counter variable
    count = 0
  
    # checks whether frames were extracted
    success = 1
  
    while success:
  
        # vidObj object calls read
        # function extract frames
        success, image = vidObj.read()
  
        # Saves the frames with frame-count
        cv2.imwrite("frame%d.jpg" % count, image)
  
        count += 1
  
  
# Driver Code
if __name__ == '__main__':
  
    # Calling the function
    FrameCapture("F:\DTU\Sem 6\IT420 Computer Vision\Colorize\test\7years.mp4")
# while cap.isOpened():
#     frame_id = int(fps*((seconds%60)*60 + seconds/60))
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#     ret, frame = video.read()
#     if not ret:
#         continue
#     cv2.imshow('frame', frame)
#     coloured = color(frame)
#     cv2.imwrite("frame%d.jpg" % count, coloured)
#     count = count + 1
#     seconds = seconds + 1
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows() 
# cv2.waitKey(0)

def color(frame):
    scaled = np.float32(frame) / 255
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 45

    print("Colorizing frame")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (frame.shape[1], frame.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")
    return colorized