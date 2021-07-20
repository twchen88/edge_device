import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

model_path = "converted_model/saved_model.xml"
arch_path = "converted_model/saved_model.bin"

net = cv2.dnn.readNet(arch_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

path_to_images = "gpu_test/notebooks/"
img = plt.imread(path_to_images + "6.jpg")
img = img[:, :, 0]
img = cv2.resize(img, (28, 28))
img = img.reshape(1, 28 * 28)/255
img = img.astype("float32")

net.setInput(img)
out = net.forward()
print(np.argmax(out, axis = 1))

# vid = cv2.VideoCapture(0)

# while True:
#     ret, frame = vid.read()

#     dat = frame[:, :, 1]
#     dat = cv2.resize(dat, (28, 28))
#     dat = dat.reshape(1, 784)

#     net.setInput(dat)
#     out = net.forward()
    
#     num = np.argmax(out, axis = 1)
#     print(num)
    
#     cv2.imshow('Input', frame)

#     # Press "ESC" key to stop webcam
#     if cv2.waitKey(1) == 27:
#         break


# # Release video capture object and close the window
# vid.release()
# cv2.destroyAllWindows()
# cv2.waitKey(1)