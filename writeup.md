# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/HOG_features.jpg
[image3]: ./output_images/windows_slide.jpg
[image4]: ./output_images/bboxes_and_heat.jpg
[image5]: ./output_images/label.jpg
[image6]: ./output_images/final_bound.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

The main solution sheets in this project are done in both, IPython Notebook `P5_solution_pipeline.ipynb` and the local python files `video_car_tracker.py`. The Ipython Notebook focusses on extracting the data, experimenting with the feature vectors and eventually training the classifier and saving it on the computer in the name of `SVM_model.p`. The file `video_car_tracker.py` utilizes the functions defined in `ImageFunctions.py` to load the classifier and modify the project video to track the cars.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is given as a function called `get_hog_features()`, which is defined in the IPython notebook `P5_solution_pipeline.ipynb` and in a separate python function file `ImageFunctions.py` from line 104 to 121.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![car_not_car][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

After trying out different HOG parameters, I tried to assign it as a feature vector for training a classifier. The classifier accuracy was the highest when the parameters were assigned as below -

`orientations = 9`
`pixels_per_cell = (8,8)`
`cells_per_block = (2,2)`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using features spatial bins of images and the histograms of the colour distribution and HOG, especially in this order. The above features from all the images of car and not-car were extracted using a function `extract_features()`, scaled using `StandardScaler()` function from `sklearn.preprocessing` library and eventually were split into training and validation datasets before being fed into the linear SVM classifier. (6th block of code from `P5_solution_pipeline.ipynb`)

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows search was implemented in two ways - the first one is when the HOG of the sub image was extracted each time and the second one when HOG of the whole image region of interest was done once and the features were extracted for all sub images. The second technique turned out to be faster than the first one. The slower first technique is implemented within the function `search_windows()` in the IPython Notebook and is executed on a random test image as shown below-

![alt text][image3]

The faster second technique is implemented as a function `find_cars()` from the function database `ImageFunctions.py` and it takes the whole image and a scale as input and returns a list of rectangular points. The scales were experimented with and seen which would give out the least false positives and took lesser time to run.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features (with the above given parameter values) plus spatially binned color and histograms of color (each with 32 bins) in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here is a [link to my video](./output_videos/project_video_tracked.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video in the form of a list of bounding boxes.  From the positive detections I created a heatmap by incrementing the region defined by the bounding boxes by 1 and then later thresholded that map to remove false positives. For getting a stable bounding box for the cars, I have used a `heatmap_history` variable to store all the heatmaps generated. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the summation of the last 15 heatmap image frames. I then assumed each blob corresponded to a vehicle and constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the result of `scipy.ndimage.measurements.label()` from a picture from the test_images folder,  and the corresponding bounding boxes then overlaid on it:

### Here is the output of `scipy.ndimage.measurements.label()` on one of the test images:
![alt text][image5]

### Here the resulting bounding boxes are drawn onto the same picture:
![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The project was laid out in a very straight forward manner, with a lot of functions which were already given in the lectures. These functions were slightly modified to suit my solution pipeline. After training the classifier, the biggest problems I faced was to reduce the false positives. The thresholding idea just wasn't enough and then I resorted to hard mine all the negative data. All the subimages that were classified as a car by the classifier were stored in folder called `positives` and all the subimages that didn't look like a car were added to the training folder of not-Car images. The classifier was trained again with these extra images and the resulting processed video showed lesser false positives.

And this is where the issue of robustness comes too. The classifier may not work well on all other kind of roads as the not-Car training folder has images mainly from this video. I haven't yet tried classifying with deep neural networks for this project, but my intuition tells me that it should work better than the linear SVM model I used here, as it is more capable of 'seeing' image features by breaking it down into smaller components and summing them up later (convolutional architecture). In this case, only the image as a feature would be used, without the computations of HOG or any other kind of histograms.
