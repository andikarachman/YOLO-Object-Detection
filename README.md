# YOLO Object Detection


## Introduction
YOLO (You Only Look Once) is a method/way to do object detection. It is the algorithm/strategy behind how the code is going to detect objects in the image. The official implementation of this idea is available through [DarkNet](https://pjreddie.com/darknet/?source=post_page---------------------------) (neural net implementation from the ground up in C from the author). It is available on [github](https://github.com/pjreddie/darknet?source=post_page---------------------------) for people to use.

Earlier detection frameworks, looked at different parts of the image multiple times at different scales and repurposed image classification technique to detect objects. This approach is slow and inefficient. YOLO takes entirely different approach. It looks at the entire image only once and goes through the network once and detects objects. Hence the name. It is very fast. That’s the reason it has got so popular. There are other popular object detection frameworks like Faster R-CNN and SSD that are also widely used.

## Pre-Requisites
We need the following files in the *yolo-coco* folder:
- *coco.names* : containing class labels our YOLO model was trained on
- *yolov3.cfg* : containing the configuration of the YOLO model
- *yolov3.weights*: containing the pre-trained weights of the YOLO model

*coco.names* and *yolov3.cfg* are already provided. 

Download *yolov3.weights* file from this [link](https://pjreddie.com/media/files/yolov3.weights) and place it in the *yolo-coco* folder or you can directly download to the *yolo-coco* folder in terminal using
 
 `$ wget https://pjreddie.com/media/files/yolov3.weights`

## Dependencies
  * opencv
  * numpy

## Command format

If the input is an image, run the following:

_$ yolo.py [-h] -i IMAGE -y YOLO [-c CONFIDENCE] [-t THRESHOLD]_

IMAGE: path to input image
YOLO: base path to YOLO directory
CONFIDENCE: minimum probability to filter weak detections
THRESHOLD: threshold when applying non-maxima suppression

If the input is a video, run the following:

_$ yolo_video.py [-h] -i INPUT -o OUTPUT -y YOLO [-c CONFIDENCE] [-t THRESHOLD]_

INPUT: path to input video
OUTPUT: path to save output video
YOLO: base path to YOLO directory
CONFIDENCE: minimum probability to filter weak detections
THRESHOLD: threshold when applying non-maxima suppression

If the input is webcam streaming, run the following:

_$ yolo_webcam.py [-h] -y YOLO [-c CONFIDENCE] [-t THRESHOLD]_ 

YOLO: base path to YOLO directory
CONFIDENCE: minimum probability to filter weak detections
THRESHOLD: threshold when applying non-maxima suppression
