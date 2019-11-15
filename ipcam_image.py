import os
import sys 
import tarfile
import tensorflow as tf 
import numpy as np 
import cv2
import datetime
import json
import urllib.request 

from playsound import playsound
from io import StringIO
from matplotlib import pyplot as plt 
from PIL import Image


sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

#Model loading
MODEL_NAME = 'person_detection_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data','label_person.pbtxt')
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def,name='')
  sess = tf.Session(graph = detection_graph)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
scores = detection_graph.get_tensor_by_name('detection_scores:0')
classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#Reading image from ipcam 
#url = "https://psychology.unl.edu/symposium/pictures/1972DonaldDJensen-RonaldRHutchinson-DanOlweus-RobertBigelow1200.jpeg"
# url = "https://psychology.unl.edu/symposium/pictures/1960RobertWWhite-FritzHeider-DavidRapaport1200.jpeg"
# url = "https://media.sgff.io/pagedata/2018-08-23/1534994898578/P&G_2a_Lean_In_Circle.jpg"
# url = "https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX4844562.jpg"
# url = "https://c1.staticflickr.com/5/4451/37798651222_684fbd9723_b.jpg"
# url = "https://ak6.picdn.net/shutterstock/videos/6854326/thumb/1.jpg"
# url = "http://footage.framepool.com/shotimg/qf/474042653-el-tovar-hotel-lobby-arizona-tourist.jpg"
url = "http://192.168.1.7/cgi-bin/snapshot.cgi"

url_response = urllib.request.urlopen(url)
img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
img = cv2.imdecode(img_array, -1)
# cv2.imshow('url image', img)

# img = cv2.imread("image1.jpg")
image_np_expanded = np.expand_dims(img, axis=0)

# Actual detection.
(boxes, scores, classes, num_detections) = sess.run(
    [boxes, scores, classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})

# Visualization of the results of a detection.
vis_util.visualize_boxes_and_labels_on_image_array(
    img,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=2)

#count time and sound
count = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
print('Number of Detected Person Is--> ', + len(count))

#Date and time
currentDT = datetime.datetime.now()
time_string = (currentDT.strftime("%Y%m%d%H%M%S"))
print('The Time Is-->')
print(time_string)

#Writing information in json
person = {'Total number of Detected Person ': len(count),"Time": int(time_string) ,"Cam Id": 27,}
with open('./json/person.json', 'w') as f:
  json.dump(person, f)

# All the results have been drawn on image. Now display the image.
#frame = cv2.resize(img, (800,500))
cv2.imshow('Human detector camera', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
