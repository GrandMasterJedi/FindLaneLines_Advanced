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
