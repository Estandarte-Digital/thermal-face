import tflite_runtime.interpreter as tflite
import cv2
import numpy as np

interpreter = tflite.Interpreter(model_path="thermal_face_automl_edge_fast.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

min_conf_threshold = .4
labels = ["label"]

video = cv2.VideoCapture('data/video.wmv')

ret, frame = video.read()

imW = frame.shape[1]
imH = frame.shape[0]

indexPhoto = 1

while ret:
    image = cv2.resize(frame, (192, 192))
    value = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], value)

    interpreter.invoke()

    # Retrieve detection results_05
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class index of detected objects
    scores = interpreter.get_tensor(output_details[2]['index'])[0]  # Confidence of detected objects
    # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * imH)))
            xmin = int(max(1, (boxes[i][1] * imW)))
            ymax = int(min(imH, (boxes[i][2] * imH)))
            xmax = int(min(imW, (boxes[i][3] * imW)))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.imwrite('results/img'+str(indexPhoto)+".png", frame)
            indexPhoto += 1

    # All the results_05 have been drawn on the image, now display the image
    #cv2.imshow('Object detector', frame)

    # Press any key to continue to next image, or press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

    ret, frame = video.read()

video.release()
cv2.destroyAllWindows()
