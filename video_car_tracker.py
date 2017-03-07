# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:41:53 2017

@author: admin
"""

from moviepy.editor import VideoFileClip
import numpy as np
import cv2, os, pickle, glob
from ImageFunctions import find_cars, 
from scipy.ndimage.measurements import label

def process_image(img):
    
    return transformed_image