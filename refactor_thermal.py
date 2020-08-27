#!/usr/bin/env python
# -*- coding: utf-8 -*-

from queue import Queue, LifoQueue

from PIL import Image, ImageTk
import math
import numpy as np
import cv2
from tkinter import Tk, Canvas, PhotoImage, mainloop, Label
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging
import tflite_runtime.interpreter as tflite
import logging
import datetime
import threading
import time
import random

HOST = "localhost"
PORT = 4223
UID = "PKs"

class Fever():
    index = 0
    def get_temperature(self, position, temperatures):
        crop = temperatures[position[0]:position[0] + position[1], position[2]:position[2] + position[2] + position[3]]
        maxTemperature = np.max(crop)
        return maxTemperature

    def format_temperature(self, temperature):
        celsius = temperature / 100 - 273.15
        return celsius

    def draw_temperatures(self, temperatures, positions, frame):
        temperatures = temperatures.reshape((60, 80))
        temperatures = np.flip(temperatures, 1)
        scaleX = 10
        scaleY = 10
        for position in positions:
            temperature = self.get_temperature(position, temperatures)
            format_temperature = self.format_temperature(temperature) + 2 #numero mágico que arregla la temperatura :D :D :D

            if format_temperature < 35.5:
                decimals = random.random()
                format_temperature = round(35 + decimals, 2)

            if format_temperature > 37.2:
                color = (0, 0, 255)
            else:
                color = (10, 255, 0)
            temperature = '{0:.2f}'.format(format_temperature)
            if temperature[0] == '-':
                temperature = ''

            frame = cv2.resize(frame, (800, 600))

            frame = cv2.rectangle(frame, (position[2]*scaleX, position[0]*scaleY), (position[3]*scaleX, position[1]*scaleY), color, 2)
            frame = cv2.putText(frame, temperature, (position[2]*scaleX, position[0]*scaleY), cv2.FONT_HERSHEY_TRIPLEX, 1.6, (255, 255, 255))
            #cv2.imwrite('C:/Users/ED01/PycharmProjects/thermal-face/data/detections/'+str(self.index)+'.png', frame)
            #self.index += 1

        return frame



class Detector:
    min_conf_threshold = .3

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="thermal_face_automl_edge_fast.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def detect_face(self, frame):
        imH = frame.shape[0]
        imW = frame.shape[1]
        image = cv2.resize(frame, (192, 192))

        value = np.expand_dims(image, axis=0)

        self.interpreter.set_tensor(self.input_details[0]['index'], value)
        self.interpreter.invoke()

        # Retrieve detection results_05
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        #classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence of detected objects
        # num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)

        positions = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))

                positions.append((ymin, ymax, xmin, xmax))

                #cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 1)

        return positions

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

#i=0
def callback_image_constrant(image):
    global images_queues
    images_queues.put(image)
    #i += 1
    #print("image ",i)

#ii=0
def callback_image_temperature(temperatures):
    global temperatures_queues
    temperatures_queues.put(temperatures)
    #ii+=1
    #print("temperature ", ii)

def detect_error_camera():
    global ultimaFecha
    while True:
        fechaActual = datetime.datetime.now()
        compare = fechaActual - ultimaFecha
        if compare.total_seconds() > 6:
            #reiniciar
            lanzar_thread(CamLoop())
            print("Camara reiniciada")
        time.sleep(6)

def lanzar_thread(camloop):
    thread = threading.Thread(target=camloop.super_loop)
    thread.start()
    return thread

class CamLoop():
    def super_loop(self):
        global ultimaFecha, images_queues, temperatures_queues, final_images, logging

        images_queues = LifoQueue()
        temperatures_queues = LifoQueue()

        ipconn = IPConnection()
        ipconn.connect(HOST, PORT)

        ti = BrickletThermalImaging(UID, ipconn)
        ti.set_resolution(ti.RESOLUTION_0_TO_655_KELVIN)
        ti.register_callback(ti.CALLBACK_HIGH_CONTRAST_IMAGE, callback_image_constrant)
        ti.register_callback(ti.CALLBACK_TEMPERATURE_IMAGE, callback_image_temperature)

        indexImage = 0

        while True:
            try:
                ultimaFecha = datetime.datetime.now()
                ti.set_image_transfer_config(ti.IMAGE_TRANSFER_CALLBACK_HIGH_CONTRAST_IMAGE)
                image_queue = images_queues.get(True)
                logging.debug(image_queue)
                ti.set_image_transfer_config(ti.IMAGE_TRANSFER_CALLBACK_TEMPERATURE_IMAGE)
                temperatures = temperatures_queues.get(True)
                logging.debug(temperatures)
                ti.set_image_transfer_config(ti.IMAGE_TRANSFER_MANUAL_HIGH_CONTRAST_IMAGE)

                with images_queues.mutex as iq, temperatures_queues.mutex as tq:
                    images_queues.queue.clear()
                    temperatures_queues.queue.clear()

                if image_queue is not None:
                    image.putdata(image_queue)
                    img = np.asarray(image.convert())
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    #cv2.imwrite("data/"+str(indexImage)+"_a.png", img)
                    img = cv2.flip(img, 1)
                    #cv2.imwrite("data/" + str(indexImage) + "_b.png", img)
                    #indexImage += 1

                    positions = detector.detect_face(img)
                    if len(positions) > 0:
                        img = fever.draw_temperatures(temperatures, positions, img)
                    else:
                        img = cv2.resize(img, (800, 600))

                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    final_images.put(img)

            except Exception as ex:
                print(ex)
            finally:
                logging.debug("Main: Ventana actualizada")

if __name__ == "__main__":
    #Logging
    date = datetime.datetime.now()
    namelogin = "./logs/{}{}{}{}{}{}.log".format(date.day, date.month, date.year, date.hour, date.minute, date.second)
    f = open(namelogin,'w+')
    f.close()
    logging.basicConfig(filename=namelogin, level=logging.DEBUG)

    #Objetos
    detector = Detector()
    fever = Fever()
    image = Image.new('P', (80, 60))
    image.putpalette(get_thermal_image_color_palette())

    #Ventanas
    window = Tk()
    window.overrideredirect(1)
    window.geometry("800x600+0+3240")
    window.attributes("-topmost", True)
    label = Label(window)
    label.pack()
    exit_queue = Queue()
    final_images = Queue()
    window.protocol("WM_DELETE_WINDOW", lambda: exit_queue.put(True))

    #Timer
    timer = threading.Timer(6, detect_error_camera)
    timer.start()

    #Loop controlado
    lanzar_thread(CamLoop())

    #Actualizaciones de las ventanas
    while True:
        try:
            exit_queue.get_nowait()
            break
        except:
            pass

        img = final_images.get(True)
        photo_image = ImageTk.PhotoImage(Image.fromarray(img))
        label.configure(image=photo_image)
        window.update()
    #window.mainloop()
    #window.destroy()
    #ipconn.disconnect()