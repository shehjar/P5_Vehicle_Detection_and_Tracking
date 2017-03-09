# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:41:53 2017

@author: admin
"""
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import numpy as np
import cv2, os, pickle, glob
from ImageFunctions import find_cars, apply_threshold, add_heat, draw_labeled_bboxes
from scipy.ndimage.measurements import label
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

curfolder = os.getcwd()
outfolder = os.path.join(curfolder,'output_videos')
if not os.path.exists(outfolder):
    os.mkdir(outfolder)

# Get classifier data
modelFileName = os.path.join(curfolder,'SVM_model.p')
model_dict = pickle.load(open(modelFileName, 'rb'))
X_scaler = model_dict['scaler']
svc = model_dict['LinearSVC']

# Image processing parameters
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 680] # Min and max in y to search in slide_window()

heatmap_history = []
def process_image(img, scale_list = [1.4,1.5]):     # (1.6, thresh = 3)
    bbox_list = []
    for scale in scale_list:
        # Find Cars in the image
        bbox_found = find_cars(img, y_start_stop[0], y_start_stop[1], scale,
                               svc, X_scaler, orient, pix_per_cell, cell_per_block, 
                               spatial_size, hist_bins)
        bbox_list.extend(bbox_found)
    heatmap = np.zeros_like(img[:,:,0])
    heatmap = add_heat(heatmap, bbox_list)
    heatmap = apply_threshold(heatmap, 2)
    heatmap_history.append(heatmap)
    sum_heatmap = np.zeros_like(heatmap)
    for heatmaps in heatmap_history[-15:]:
        sum_heatmap += heatmaps
    #sum_heatmap = heatmap
    labels = label(sum_heatmap)
    transformed_image = draw_labeled_bboxes(img, labels)
    return transformed_image

input_video = 'project_video.mp4'
output_filename = input_video.split('.mp4')[0] + '_tracked.mp4'
output_video = os.path.join(outfolder,output_filename)

clip = VideoFileClip(input_video)
video_clip = clip.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)

# Below lines of code were used for Debugging purposes only.
#label_values = []
#for heatmap in heatmap_history:
#    labels = label(heatmap)
#    label_values.append(labels[1])
#label_values = np.array(label_values)
## Get maximum labelled frame
#id_label = np.argmax(label_values) #1035
#heatmap = heatmap_history[id_label]
#val = np.min(heatmap[heatmap.nonzero()])