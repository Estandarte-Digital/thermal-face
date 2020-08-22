#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import cv2

archivos = glob.glob("./data/reescalado/*.png")

for archivo in archivos:
    img = cv2.imread(archivo)
    img = cv2.resize(img, (192, 192))
    cv2.imshow("img", img)
    cv2.waitKey(0)