import cv2
import sys

from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QLabel, QApplication,QMainWindow
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import math
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import time

global ipconIMG, ipconTMP, HOST, PORT, UID, WIDTH, HEIGHT, SCALE

HOST = "localhost"
PORT = 4223
UID = "PKs" # Change XYZ to the UID of your Thermal Imaging Bricklet
WIDTH = 80
HEIGHT = 60
SCALE = 5

def get_thermal_image_color_palette():
    palette = []

    for x in range(256):
        x /= 255.0
        palette.append(int(round(255 * math.sqrt(x))))  # RED
        palette.append(int(round(255 * pow(x, 3))))  # GREEN
        if math.sin(2 * math.pi * x) >= 0:
            palette.append(int(round(255 * math.sin(2 * math.pi * x))))  # BLUE
        else:
            palette.append(0)

    return palette

def cb_high_contrast_image(image):
    # Save image to queue (for loop below)
    global image_queue
    image_queue.put(image)

class Fever:
    def get_temperature(self, temperatures, bbox):
        # Consider the raw temperatures insides the face bounding box.
        crop = temperatures[bbox['top']:bbox['bottom'], bbox['left']:bbox['right']]
        if crop.size == 0:
            return None

        # Use the maximum temperature across the face.
        return np.max(crop), crop

    def format_temperature(self, temperature):
        celsius = temperature / 100 - 273.15
        return celsius

    def draw_temperatures(self, temperatures, faces, image, scale_x=1, scale_y=1):
        for face in faces:
            box = {
                'left': int(face[0, 0]),
                'top': int(face[0, 1]),
                'right': int(face[1, 0]),
                'bottom': int(face[1, 1])
            }
            temperature, crop = self.get_temperature(temperatures, box)
            format_temperature = self.format_temperature(temperature)
            # ymin = int(max(1, (boxes[i][0] * imH)))
            # xmin = int(max(1, (boxes[i][1] * imW)))
            # ymax = int(min(imH, (boxes[i][2] * imH)))
            # xmax = int(min(imW, (boxes[i][3] * imW)))

            # cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
            cv2.putText(image, format_temperature, (box['left'],box['top']-20), cv2.FONT_HERSHEY_COMPLEX, 5, (255, 255, 255))

        return image


class Detector:
    min_conf_threshold = .5

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="thermal_face_automl_edge_fast.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect_face(self, frame):
        #imH = frame.shape[0]
        #imW = frame.shape[1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(frame, (192, 192))

        value = np.expand_dims(image, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], value)
        self.interpreter.invoke()

        # Retrieve detection results_05
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        #classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence of detected objects
        # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        faces = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                faces.append(boxes)
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                #ymin = int(max(1, (boxes[i][0] * imH)))
                #xmin = int(max(1, (boxes[i][1] * imW)))
                #ymax = int(min(imH, (boxes[i][2] * imH)))
                #xmax = int(min(imW, (boxes[i][3] * imW)))

                #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        return frame, faces

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        #cap = cv2.VideoCapture(0)
        detector = Detector()
        fever = Fever()
        image = Image.new('P', (WIDTH, HEIGHT))
        image.putpalette(get_thermal_image_color_palette())
        image = image.resize((WIDTH, HEIGHT))
        ti = BrickletThermalImaging(UID, ipconIMG)

        while True:
            try:
                ti.set_image_transfer_config(ti.IMAGE_TRANSFER_MANUAL_HIGH_CONTRAST_IMAGE)
                constrast_image = ti.get_high_contrast_image()
                image.resize((80, 60))
                image.putdata(constrast_image)
                img = np.asarray(image.convert())
                img, boxes = detector.detect_face(img)

                ti.set_image_transfer_config(ti.IMAGE_TRANSFER_MANUAL_TEMPERATURE_IMAGE)
                temperatures = ti.get_temperature_image()

                img = fever.draw_temperatures(temperatures, boxes, img)

                #cv2.imshow("thermal", img)
                #cv2.waitKey(1)

                h, w, ch = img.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(img, w, h, bytesPerLine, QImage.Format_BGR888)
                p = convertToQtFormat.scaled(640, 480, Qt.IgnoreAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(.25)
            except Exception as ex:
                print(ex)



class App(QMainWindow):
    def __init__(self):
        super().__init__()
        [...]
        self.initUI()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == QtCore.Qt.Key_Q:
            self.close()

    def initUI(self):
        #self.setWindowTitle(self.title)
        #self.setGeometry(self.left, self.top, self.width, self.height)
        self.setWindowFlags(Qt.FramelessWindowHint)
        #self.resize(1800, 1200)
        self.resize(640, 480)
        # create a label
        self.label = QLabel(self)
        self.label.move(0, 0)
        self.label.resize(640, 480)
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
        self.show()

if __name__ == "__main__":
    retry = True

    while retry:
        try:
            ipconIMG = IPConnection()  # Create IP connection
            ipconTMP = IPConnection()

            ipconIMG.connect(HOST, PORT)  # Connect to brickd
            ipconTMP.connect(HOST, PORT)
            # Don't use device before ipcon is connected

            app = QApplication(sys.argv)
            window = App()
            app.exec_()
            retry = False
        except Exception as ex:
            print(ex)

    ipconIMG.disconnect()
    ipconTMP.disconnect()
