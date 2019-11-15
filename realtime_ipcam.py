import os
import sys 
import tarfile
import tensorflow as tf 
import numpy as np 
import cv2
import urllib.request
import time
import json

from io import StringIO
from matplotlib import pyplot as plt 
from PIL import Image
from playsound import playsound

cap = cv2.VideoCapture('http://192.168.0.101:8080/video')

sys.path.append("..")

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
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

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
	(im_width,im_height) = image.size 
	return np.array(image.getdata()).reshape((im_height,im_width,3)).astype(np.uint8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')

      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      #  Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),

          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)

      #count time and sound 
      count = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
      if(len(count)) >= 1:
        print(len(count))
        tm = time.localtime() # get struct_time
        time_string = time.strftime("%m%d%Y%H%M%S", tm)
        print(time_string)
        person = {'Total number of Detected Person ': len(count),"Time": int(time_string) ,"Cam Id": 27}
        with open('./json/person.json', 'w') as f:
        	json.dump(person,f)

        # playsound('sound2.mp3')

      cv2.imshow('HD cam', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
