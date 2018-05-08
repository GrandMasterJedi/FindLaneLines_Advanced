[//]: # (Image References)
[image1]: ./example/ChessboardCornersCalibration.png "ChessCalibration"
[image2]: ./example/UndistortedChessboard.png "ChessUndistorted"
[image3]: ./example/UndistortedLane.png "UndistortedLane"
[image4]: ./example/ColorGradientFilter.png "ComboGrad"
[image5]: ./example/WarpedLane.png "WarpLane"
[image6]: ./example/LanesLineFit2.png "LaneLineFit"
[image7]: ./output_images/straight_lines1.jpg "outImage1"
[image8]: ./output_images/test1.jpg "outImage2"
[video1]: ./videoOutput1.mp4 "Video"


# **Advanced Lane Finding**

Assuming a camera is mounted on the car capturing images of the road lane ahead, I detect lane lines using advanced computer vision techniques that include image distortion correction, perspective transformation and gradient filtering.

The pipeline I provide is able to correctly identify lane lines for an _easy_ video clip, taken on relatively simple road conditions.

The file *main.ipynb* contains all the program code

---
## Dependencies
* Python 3.x
* NumPy
* OpenCV
* Matplot.pyplot
* MoviePy
* glob
* pickle
* OS

---
## Goals / Steps
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Rubric Points 
---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration is done using pictures of a chessboard hanged on the wall taken from different perspectives. I use pre-build calibration function in OpenCV that recognize inner corners of each square in the chessboard and record their coordinates in the images that are contrasted with real world coordinate for an undistorted chessboard.

Example of the detected squares in the chessboard images are highlighted in the image below.
![alt text][image1]

The real world coordinates (x, y, z) are contained in the `objpoints` construct. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. The construct `imgpoints`, instead contains the (x, y) pixel position of each of the corners in the distorted image plane with each successful chessboard detection.  

The constructs `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients using the 
`cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
![alt text][image2]


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
The distortion coefficients _dist_ and the camera calibration matrix _mtx_ are then used to undistort images of the road lane taken with the camera mounted on the car.

Below is an example of a distortion corrected image of the lane:

![alt text][image3]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  

After the image undistortion, a combined filter on the image gradient and HLS color transform is applied. The combined filter is defined in *combinedThresh(img)*, contained in cell 12 of the _main.ipynb_. More specifically, the combined filter consist of an absolute Sobel threshold x-orientation filter (*absGr*), a gradient magnitude filter (*magGr*), a filter on the direction of the gradient (*dirGr*) and the filter on HLS color (*hlsGr*). The threshold parameter are set as below. Other parameter could be experimented to improve the lane detection pipelane.

```python
    absGr = absSobelThresh(img, orient='x', thresh_min=50, thresh_max=255)
    magGr = magThresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dirGr = dirThresh(img, sobel_kernel=15, thresh=(0.7, 1.3))
    hlsGr = hls_thresh(img, thresh=(170, 255))
```	
	
The following is an example of the output of the different filters applied to the test image.
![alt text][image4]


#### 3. Describe how you performed a perspective transform and provide an example of a transformed image.

The perspective transform is defined in *perspectiveTransform(img)*, contained in cell 4 of _main.ipynb_. The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
    dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
```

The _src_ coordinates correspond to the highlighted area in the left image below (original), while the _dst_ coordinates refers to the corresponding coordinates to the _src_ in the right image below (warped)

![alt text][image5]


#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

After the image transformation, I identify lane line pixels by fitting a 2nd order polynomial to different points for respectively left and right lane lines. The lane line identification function is defined in *lineFit(binary_warped)*, cell 14 of main.ipynb.
The algorithms is relatively complex. It consists of finding midpoint of rectangles (sliding windows) build around potential lane lines. The image below shows the result of the lane line fitting.

![alt text][image6]

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature and the position of the vehicle with respect to the lane center is calculated in definition *curveRadius_CenterDist(binImg, ret)* in cell 24 of _main.ipynb_. (Click [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php) for an awesome tutorial on how to calculate the radius.)

The radius of the curvature is converted from pixel image coordinates to meters of the real world assuming 30 meters for each 720 vertical pixels (ym_per_pix = 30/720). The position of the vehicle with respect to the identified lane is also calculated as the difference of the x-midpoint of the image with respect to the x-coordinate of the lane center and converted to meters assuming 3.7 meters (lane width) every 690 horizontal pixels of the image (xm_per_pix = 3.7/690)


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The folder (output_images)[./output_images/] contains identified lane with curvature and vehicle position estimation. The examples below shows correct identification of the lane and correct estimation of curvature radius and vehicle position.

![alt text][image7]

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Check out the video [here](./videoOutput1.mp4), or download and open with your computer video reader _videoOutput1.mp4_

## Discussion
1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?


The lane recognition pipeline succeded to detect road lanes in the video clip. However the conditions for the road and image colors were relatively simple. The same pipeline failed with videos with more challenging road and color conditions.

Overall, the presence of elements such as shadow, road sign placed within the lane, or road paved with different colors are detrimental to the success of the pipeline. To counter this shortcoming, one can play with different color transform, such as LUV transform, and different color and gradient thresholding. In my pipeline, I used _reasonable_ values, without trying many thresholding.

For future work, a sanity check can improve the lane detection especially for challenging videos. 
As one anonymous project reviewer suggested, the sanity check can answer the following points:
* Are the two polynomials an appropriate distance apart based on the known width of a highway lane?
* Do the two polynomials have same or similar curvature?
* Have these detections deviated significantly from those in recent frames?

When the sanity check fails on one image frame, the lane detection result can be discarded and a proxy result, such as the previous lane detection result, can be taken.




---
## Resources
* Udacity project assignment and template on [GitHub](https://github.com/udacity/CarND-Advanced-Lane-Lines)
* Udacity project [rubric](https://review.udacity.com/#!/rubrics/571/view) points

