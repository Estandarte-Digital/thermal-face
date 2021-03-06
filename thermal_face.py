#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np
from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging

HOST = "localhost"
PORT = 4223
UID = "PKs" # Change XYZ to the UID of your Thermal Imaging Bricklet

from tinkerforge.ip_connection import IPConnection
from tinkerforge.bricklet_thermal_imaging import BrickletThermalImaging

import math
import time
import cv2
import numpy as np

from tkinter import Tk, Canvas, PhotoImage, mainloop, Label  # Python 3
from queue import Queue, Empty

from PIL import Image, ImageTk

WIDTH = 80
HEIGHT = 60
SCALE = 5  # Use scale 5 for 400x300 window size (change for different size). Use scale -1 for maximized mode

image_queue = Queue()


# Creates standard thermal image color palette (blue=cold, red=hot)
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


# Callback function for high contrast image
def cb_high_contrast_image(image):
    # Save image to queue (for loop below)
    global image_queue
    image_queue.put(image)


def on_closing(window, exit_queue):
    exit_queue.put(True)


if __name__ == "__main__":
    ipcon = IPConnection()  # Create IP connection
    ti = BrickletThermalImaging(UID, ipcon)  # Create device object

    ipcon.connect(HOST, PORT)  # Connect to brickd
    # Don't use device before ipcon is connected

    # Register illuminance callback to function cb_high_contrast_image
    ti.register_callback(ti.CALLBACK_HIGH_CONTRAST_IMAGE, cb_high_contrast_image)

    # Enable high contrast image transfer for callback
    ti.set_image_transfer_config(ti.IMAGE_TRANSFER_CALLBACK_HIGH_CONTRAST_IMAGE)

    # Create Tk window and label
    window = Tk()

    # Run maximized
    if SCALE == -1:
        window.geometry("%dx%d+0+0" % (window.winfo_screenwidth(), window.winfo_screenheight()))
        window.update()  # Update to resize the window

        w, h = window.winfo_width(), window.winfo_height()
        SCALE = min(w // WIDTH, h // HEIGHT)

    label = Label(window)
    label.pack()

    image = Image.new('P', (WIDTH, HEIGHT))
    # This puts a color palette into place, if you
    # remove this line you will get a greyscale image.
    image.putpalette(get_thermal_image_color_palette())

    exit_queue = Queue()
    window.protocol("WM_DELETE_WINDOW", lambda: on_closing(window, exit_queue))

    while True:
        try:
            exit_queue.get_nowait()
            break  # If the exit_queue is not empty, the window was closed.
        except Empty:
            pass

        # Get image from queue (blocks as long as no data available)
        image_data = image_queue.get(True)

        # Use original width/height to put data and resize again afterwards
        image = image.resize((WIDTH, HEIGHT))
        image.putdata(image_data)
        image = image.resize((WIDTH * SCALE, HEIGHT * SCALE), Image.ANTIALIAS)
        img = np.asarray(image.convert())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)

        cv2.waitKey(1)

        # Translate PIL Image to Tk PhotoImageShow and show as label
        #photo_image = ImageTk.PhotoImage(image)
        #label.configure(image=photo_image)

        #window.update()

    window.destroy()