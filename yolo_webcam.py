# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
WEIGHTS_PATH = os.path.sep.join([args["yolo"], "yolov3.weights"])
CONFIG_PATH = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load confidence and threshold level
CONFIDENCE = args["confidence"]
THRESHOLD = args["threshold"]


def load_run_model(frame, convert_to_rgb=True):

	# load our YOLO object detector trained on COCO dataset (80 classes)
	net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)

	# determine only the *output* layer names that we need from YOLO
	ln = net.getLayerNames()
	ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes and
	# associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=convert_to_rgb, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	return layerOutputs


def generate_bounding_boxes(layerOutputs, W, H):

	# initialize our lists of detected bounding boxes, confidences, and
	# class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability) of
			# the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > CONFIDENCE:
				# scale the bounding box coordinates back relative to the
				# size of the frame, keeping in mind that YOLO actually
				# returns the center (x, y)-coordinates of the bounding
				# box followed by the boxes' width and height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top and
				# and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates, confidences,
				# and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	
	return boxes, confidences, classIDs


def paint_box_on_object(frame, box, color, label, confidence):

	# unpack the coordinates of the box
	x, y, w, h = box[0], box[1], box[2], box[3]

	# draw a bounding box rectangle and label on the frame
	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
	text = "{}: {:.4f}".format(label, confidence)
	cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_object_detection():
	
	# define our input 
	video_capture = cv2.VideoCapture(0)

	print("[INFO] running object detection...")

	# loop over frames from the video file stream
	while video_capture.isOpened():

		# read the next frame from the file
		(ret, frame) = video_capture.read()
		frame = cv2.resize(frame, None, fx=0.5, fy=0.5)

		# if the frame was not grabbed, then we have reached the end
		# of the stream
		if not ret:
			break

		# grab the frame dimensions
		H, W = frame.shape[:2]

		# perform a forward pass of the YOLO object detector
		layerOutputs = load_run_model(frame)

		# generate detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes, confidences, classIDs = generate_bounding_boxes(layerOutputs, W, H)

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():

				color = [int(c) for c in COLORS[classIDs[i]]]
				label = LABELS[classIDs[i]]
				confidence = confidences[i]
				
				# draw a bounding box rectangle and label on the frame
				paint_box_on_object(frame, boxes[i], color, label, confidence)

		# Display the resulting image
		cv2.imshow('frame', frame)
		
		# Hit 'q' on the keyboard to stop the loop
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	# release the file pointers
	print("[INFO] cleaning up...")
	video_capture.release()
	cv2.destroyAllWindows()


# Run object detection
run_object_detection()

