import cv2
from tensorflow import keras
import numpy as np


# load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# load IR model and specify NCS
model_path = "converted_model/saved_model.xml"
arch_path = "converted_model/saved_model.bin"
net = cv2.dnn.readNet(arch_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# use these data to test
images = x_test.reshape(10000, 28 * 28)/255
images = images.astype("float32")
labels = y_test
correct = 0

# test all of them
for i in range(10000):
    img = images[i]
    net.setInput(img)
    out = net.forward()
    if np.argmax(out, axis = 1) == labels[i]:
        correct += 1

print(correct/10000)
