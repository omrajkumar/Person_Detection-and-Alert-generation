import os 
import tensorflow as tf 
import numpy as np 
import cv2
import datetime
import json
import time
import urllib.request
import psutil 
# import sys

from playsound import playsound
# from io import StringIO
from matplotlib import pyplot as plt 
# from PIL import Image


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


with open('camaddress.txt', 'r') as f:
  cam = f.readlines()
size = len(cam)

#Model loading
MODEL_NAME = 'person_detection_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data','label_person.pbtxt')
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 2)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict
# url_response = ['url_response0','url_response1','url_response2','url_response3','url_response4','url_response5']
# img_array = ['img_array0','img_array1','img_array2','img_array3','img_array4','img_array5']
# img = ['img0','img1','img2','img3','img4','img5']
# num = len(cam)
while True:
    for i in range(0,num):
      n = str(i)
      url_response.append('url_response'+n)
      img_array.append('img_array'+n)
      img.append('img'+n)    
      url_response[i] = urllib.request.urlopen(cam[i])
      img_array[i] = np.array(bytearray(url_response[i].read()), dtype=np.uint8)
      img[i] = cv2.imdecode(img_array[i], -1)
  # Actual detection.
      output_dict = run_inference_for_single_image(img[i], detection_graph)
  # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
        img[i],
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=2)
    # #Date and time
    currentDT = datetime.datetime.now()
    time_string = (currentDT.strftime("%Y%m%d%H%M%S"))
    print('The Time Is-->')
    print(time_string)
    cpu=psutil.cpu_percent()
    ram= psutil.virtual_memory()
    print(cpu)
    print(ram)
    imgout1 = np.concatenate((img[0],img[1],img[2]), axis=1)
    imgout2 = np.concatenate((img[3],img[4],img[5]), axis=1)
    finelout = np.concatenate((imgout1,imgout2), axis=0)
    cv2.imshow('HDcam',cv2.resize(finelout, (1200,600)))


    time.sleep(0.5)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
        cv2.destroyAllWindows()
  
