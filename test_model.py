import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

interpreter = tflite.Interpreter(model_path="thermal_face_automl_edge_fast.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

original = cv2.imread("data/pic04.jpg")
image = cv2.resize(original, (192, 192))

value = np.expand_dims(image, axis=0)

interpreter.set_tensor(input_details[0]['index'], value)

interpreter.invoke()

#output_data = interpreter.get_tensor(output_details[0]['index'])

# Retrieve detection results_05
boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
# num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

min_conf_threshold = .5
imH = original.shape[1]
imW = original.shape[0]
labels = ["label"]

# Loop over all detections and draw detection box if confidence is above minimum threshold
for i in range(len(scores)):
    if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
        # Get bounding box coordinates and draw box
        # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
        ymin = int(max(1, (boxes[i][0] * imH)))
        xmin = int(max(1, (boxes[i][1] * imW)))
        ymax = int(min(imH, (boxes[i][2] * imH)))
        xmax = int(min(imW, (boxes[i][3] * imW)))

        cv2.rectangle(original, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
        cv2.imwrite("results_05/pic04.jpg", original)

# All the results_05 have been drawn on the image, now display the image
cv2.imshow('Object detector', original)

# Press any key to continue to next image, or press 'q' to quit
if cv2.waitKey(0) == ord('q'):
    cv2.destroyAllWindows()
