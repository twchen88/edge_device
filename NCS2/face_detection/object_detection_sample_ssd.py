import cv2
import sys, os

model_path = "face-detection-retail-0004.xml"
arch_path = "face-detection-retail-0004.bin"

net = cv2.dnn.readNet(arch_path, model_path)

vid = cv2.VideoCapture(0)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

while True:
    ret, frame = vid.read()

    frame = cv2.flip(frame, 1)

    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), ddepth=cv2.CV_8U)
    net.setInput(blob)
    out = net.forward()
    
    # Draw detected faces
    for detect in out.reshape(-1, 7):
        conf = float(detect[2])
        xmin = int(detect[3] * frame.shape[1])
        ymin = int(detect[4] * frame.shape[0])
        xmax = int(detect[5] * frame.shape[1])
        ymax = int(detect[6] * frame.shape[0])

        if conf > 0.5:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 0), thickness=2)
    
    cv2.imshow('Input', frame)
    
    # Press "ESC" key to stop webcam
    if cv2.waitKey(1) == 27:
        break

# Release video capture object and close the window
vid.release()
cv2.destroyAllWindows()
cv2.waitKey(1)