# Installing Libraries
!pip install tensorflow opencv-python matplotlib
# Importing important libraries for image processing and model usage
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
#Loading a pre-trained model 
model = tf.saved_model.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
#Reading input image
image = cv2.imread('/content/test.jpg')
#Converting image from BGR to RGB Format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#Converting image to tensor format
input_tensor = tf.convert_to_tensor(image_rgb)
input_tensor = input_tensor[tf.newaxis, ...]
detections = model(input_tensor)
# Looping through top 5 detections and drawing bounding boxes
for i in range(5):
    box = detections['detection_boxes'][0][i].numpy()
    score = detections['detection_scores'][0][i].numpy()
     # Only showing boxes if confidence score is greater than 50%
    if score > 0.5:
        h, w, _ = image.shape
        y1, x1, y2, x2 = box
        # Drawing rectangle on detected object
        cv2.rectangle(image, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (0,255,0), 2)
#Displaying Final output
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')