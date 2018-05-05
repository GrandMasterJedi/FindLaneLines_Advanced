def camCalibrate(imagesPath, chessInnerCorner_x, chessInnerCorner_y, save = False, plot = True):
    """
    Calibrate the camera used to take pictures of the chessboard appended on the wall. All the images must be of the same size
    imagesPath:          path of the images containing chessboard picture
    chessInnerCorner_h:   number of inner corners on x axis (horizontal)
    chessInnerCorner_y:    number of inner corners on y axis (vertical)
    """
    totCorners = chessInnerCorner_x*chessInnerCorner_y
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) to store the chessboard square coordinate
    objp = np.zeros((totCorners,3), np.float32)
    objp[:,:2] = np.mgrid[0:chessInnerCorner_x, 0:chessInnerCorner_y].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    if plot == True:
        nimg = len(images)
        nr = int(nimg/4)
        fig, axs = plt.subplots(nr,4, figsize=(15, 10))
        axs = axs.ravel() 
    else:
        fig = [];


    for idx, fname in enumerate(imagesPath):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chessInnerCorner_x, chessInnerCorner_y), None)

        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            
        if plot == True:
            img = cv2.drawChessboardCorners(img, (chessInnerCorner_x, chessInnerCorner_y), corners, ret)
            axs[idx].imshow(img)
                

    # Get the size of on image
    timg = cv2.imread(imagesPath[0])
    imgSize = (timg.shape[1], timg.shape[0])
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgSize, None, None)
    
    if (save ==True):
        calibrationData = {}
        calibrationData["mtx"] = mtx
        calibrationData["dist"] = dist
        pickle.dump(calibrationData, open("resource/calibration.p", "wb" ) )
        
    return mtx, dist, fig
	
	
def perspectiveTransform(img):
    """
    Transform the perspective of the image captured by camera mounted on the car
    Note: it may be neccessary to cange the source and destination coordinates
    
    warped:  image transformed from source to destination
    unwarp:  image transformed from destination to source
    """

    imgSize = (img.shape[1], img.shape[0])
    
    # Source and destination points for transform. Points coordinates approx
    # [lowLeft, lowright, upLeft, upRight]
    src = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
    dst = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])

    m = cv2.getPerspectiveTransform(src, dst)
    mInv = cv2.getPerspectiveTransform(dst, src)

    warp = cv2.warpPerspective(img, m, imgSize, flags=cv2.INTER_LINEAR)
    unwarp = cv2.warpPerspective(warp, mInv, (warp.shape[1], warp.shape[0]), flags=cv2.INTER_LINEAR)  

    return m, mInv, warp, unwarp

	
	
def absSobelThresh(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Identify pixels of an image where the gradient falls within a specified threshold range
    arguments:  img, 
                gradient orientation: "x", or "y:
                and threshold min/max values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        absSobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        absSobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaledSobel = np.uint8(255*absSobel/np.max(absSobel))

    # Create a copy and apply the threshold
    binaryOutput = np.zeros_like(scaledSobel)
    binaryOutput[(scaledSobel >= thresh_min) & (scaledSobel <= thresh_max)] = 1

    # Return the result
    return binaryOutput

def magThresh(img, sobel_kernel=3, mag_thresh=(30, 100)):
    """
    Return the magnitude of the gradient for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradMag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scaleFactor = np.max(gradMag)/255
    gradMag = (gradMag/scaleFactor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binaryOutput = np.zeros_like(gradMag)
    binaryOutput[(gradMag >= mag_thresh[0]) & (gradMag <= mag_thresh[1])] = 1

    # Return the binary image
    return binaryOutput


def dirThresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Return the direction of the gradient for a given sobel kernel size and threshold values
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction, apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binaryOutput =  np.zeros_like(absgraddir)
    binaryOutput[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binaryOutput


def hls_thresh(img, thresh=(100, 255)):
    """
    Convert RGB to HLS and threshold to binary image using S channel
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binaryOutput = np.zeros_like(s_channel)
    binaryOutput[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binaryOutput

	
	
def combinedThresh(img):
    """
    """
    absGr = absSobelThresh(img, orient='x', thresh_min=50, thresh_max=255)
    magGr = magThresh(img, sobel_kernel=3, mag_thresh=(50, 255))
    dirGr = dirThresh(img, sobel_kernel=15, thresh=(0.7, 1.3))
    hlsGr = hls_thresh(img, thresh=(170, 255))

    combined = np.zeros_like(dirGr)
    combined[(absGr == 1 | ((magGr == 1) & (dirGr == 1))) | hlsGr == 1] = 1

    return combined, absGr, magGr, dirGr, hlsGr 
	

	
def lineFit(binary_warped):
    """
    Find and fit lane lines to a warped image
    binary_warped:     binary image that has been previously warped for lane line
    
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

    # Create an output image to draw on and visualize the result
    outImg = (np.dstack((binary_warped, binary_warped, binary_warped))*255).astype('uint8')

    # Find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[100:midpoint]) + 100
    rightx_base = np.argmax(histogram[midpoint:-100]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    margin = 100  # Set the width of the windows +/- margin  
    minpix = 50   # Set minimum number of pixels found to recenter window

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    rectangle = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(outImg,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(outImg,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        rectangle.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['outImg'] = outImg
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret, histogram, rectangle, outImg
	
	
	
def visualizeFit(tImg, ret):
    """
    tImg:
    ret:  leftFit, rightFit:
    """
    
    # Grab variables from ret dictionary
    leftFit = ret['left_fit']
    rightFit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    leftLaneInds = ret['left_lane_inds']
    rightLaneInds = ret['right_lane_inds']
  
    h = tImg.shape[0]
    leftFit_x_int = leftFit[0]*h**2 + leftFit[1]*h + leftFit[2]
    rightFit_x_int = rightFit[0]*h**2 + rightFit[1]*h + rightFit[2]
    #print('fit x-intercepts:', left_fit_x_int, right_fit_x_int)

    # Create an output image to draw on and  visualize the result
    outImg2 = np.uint8(np.dstack((tImg, tImg, tImg))*255)

    # Generate x and y values for plotting
    ploty = np.linspace(0, tImg.shape[0]-1, tImg.shape[0] )
    leftFitx = leftFit[0]*ploty**2 + leftFit[1]*ploty + leftFit[2]
    rightFitx = rightFit[0]*ploty**2 + rightFit[1]*ploty + rightFit[2]

    for r in rect: 
        cv2.rectangle(outImg2,(r[2], r[0]),(r[3],r[1]),(0,255,0), 2) 
        cv2.rectangle(outImg2,(r[4], r[0]),(r[5],r[1]),(0,255,0), 2)

    # Identify the x and y positions of all nonzero pixels in the image
#     nonzero = tImg.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
    outImg2[nonzeroy[leftLaneInds], nonzerox[leftLaneInds]] = [255, 0, 0]
    outImg2[nonzeroy[rightLaneInds], nonzerox[rightLaneInds]] = [100, 200, 255]
    plt.imshow(outImg2)
    plt.plot(leftFitx, ploty, color='orange', linewidth = 4)
    plt.plot(rightFitx, ploty, color='orange', linewidth =4)
    plt.title('Line Fit', fontsize=20)

	
	
def continueLineFit(binary_warped, left_fit, right_fit):
    """
    Given a previously fit line, quickly try to find the line based on previous lines
    binary_warped:         new binary image, previously warped to detect lanes
    left_fit, right_fit:   fitted line from previous warped image
    """
    # Assume you now have a new warped binary image from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # If we don't find enough relevant points, return all None (this means error)
    min_inds = 10
    if lefty.shape[0] < min_inds or righty.shape[0] < min_inds:
        return None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Return a dict of relevant variables
    ret = {}
    ret['left_fit'] = left_fit
    ret['right_fit'] = right_fit
    ret['nonzerox'] = nonzerox
    ret['nonzeroy'] = nonzeroy
    ret['left_lane_inds'] = left_lane_inds
    ret['right_lane_inds'] = right_lane_inds

    return ret
	
	
	
## Calculate curvature
def curveRadius_CenterDist(binImg, ret):
    """
    Calculate radius of curvature in meters
    Define y-value where we want radius of curvature. I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    Formula for left and right curvature radius:
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    """
    
    # Grab variables from ret dictionary
#     leftFit = ret['left_fit']
#     rightFit = ret['right_fit']
    nonzerox = ret['nonzerox']
    nonzeroy = ret['nonzeroy']
    leftLaneInds = ret['left_lane_inds']
    rightLaneInds = ret['right_lane_inds']
    
#     leftCurveRad, rightCurveRad, centerDist = (0, 0, 0)
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    h = binImg.shape[0]
    ploty = np.linspace(0, h-1, h)
    y_eval = np.max(ploty)
    # y_eval = 719  # correspond to the lowest y coordinate of a 720p image

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720  # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Identify the x and y positions of all nonzero pixels in the image
#     nonzero = binImg.nonzero()
#     nonzeroy = np.array(nonzero[0])
#     nonzerox = np.array(nonzero[1])
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    
    

    # Fit new polynomials to x,y under real meters measure
    leftFit = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    rightFit = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    
    # Calculate the new curvature radius in meters
    leftCurveRad = ((1 + (2*leftFit[0]*y_eval*ym_per_pix + leftFit[1])**2)**1.5) / np.absolute(2*leftFit[0])
    rightCurveRad = ((1 + (2*rightFit[0]*y_eval*ym_per_pix + rightFit[1])**2)**1.5) / np.absolute(2*rightFit[0])
    
    # Distance from center is image x midpoint - mean of leftFit and rightFit intercepts 
    if rightFit is not None and leftFit is not None:
        tposition = binImg.shape[1]/2  # horizontal midpoint 
        lFitx = leftFit[0]*h**2 + leftFit[1]*h + leftFit[2]
        rFitx = rightFit[0]*h**2 + rightFit[1]*h + rightFit[2]
        laneCenter = (rFitx + lFitx) /2
        centerDist = (tposition - laneCenter) * xm_per_pix
             
    return leftCurveRad, rightCurveRad, centerDist
	
	
def visualizeLaneOverlay(origImg, warpImg, leftFit, rightFit, mInv, lRad, rRad, cDist):
    """
    Final lane line prediction visualized and overlayed on top of original image
    origImage:     original image of the lane (h0, w0, 3)
    warpImage:     binary warped image (h,w)
    """
    # Generate x and y values for plotting
    ploty = np.linspace(0, origImg.shape[0]-1, origImg.shape[0])
    
    leftFit2 = leftFit[0]*ploty**2 + leftFit[1]*ploty + leftFit[2]
    rightFit2 = rightFit[0]*ploty**2 + rightFit[1]*ploty + rightFit[2]

    h,w = warpImg.shape
    # Create an image to draw the lines on
    warpZero = np.zeros_like(warpImg).astype(np.uint8)
    colorWarp = np.dstack((warpZero, warpZero, warpZero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(colorWarp, np.int_([pts]), (0,255, 0))
    cv2.polylines(colorWarp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(colorWarp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newWarp = cv2.warpPerspective(colorWarp, mInv, (w, h))
    
    # Combine the result with the original image
    result = cv2.addWeighted(origImg, 1, newWarp, 0.3, 0)

    # Annotate lane curvature values and vehicle offset from center
    avgRadCurve = (lRad + rRad)/2
    labRad = 'Radius of curvature:       %.1f m' % avgRadCurve
    result = cv2.putText(result, labRad, (20,50), 0, 1, (0,0,0), 2, cv2.LINE_AA)

    labDist = 'Distance from lane center: %.1f m' % cDist
    result = cv2.putText(result, labDist, (20,100), 0, 1, (0,0,0), 2, cv2.LINE_AA)    

    return result

	
