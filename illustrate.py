#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:38:57 2017

@author: Hannah Holman (hholman - R01118537)

On my honor, I have not given, nor recieved, nor witnessed any unathorized 
assistance on this work.

I worked on this project alone, and referred to the following resources:
https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_grabcut/py_grabcut.html#grabcut
https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
https://docs.opencv.org/3.3.0/dd/d49/tutorial_py_contour_features.html
http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
http://www.askaswiss.com/2016/01/how-to-create-cartoon-effect-opencv-python.html
https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
https://www.learnopencv.com/tag/pencilsketch/
"""
import numpy as np
import cv2
    
def quantize_color(image):
    """ This function performs a quanization of an image, reducing color variation
    
        Args:
            image (numpy.ndarray): the image to be quantized
        Returns:
            res2 (numpy.ndarray): the quantized image
    """
    
    img = image.copy()
    
    #reshape image into imagePixelsx3 size
    img_sample = img.reshape((-1, 3))
    img_sample = np.float32(img_sample)
    
    #define critieria for quantization with type of criteria, max num of 
    #iterations, and required level of accuracy (epsilon)
    
    #SWANS
    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    #number of clusters/colors to be represented in output image
    k=8
    
    #apply kmeans 
    ret, label, center = cv2.kmeans(img_sample, k, None, criteria, 10, 
                                    cv2.KMEANS_RANDOM_CENTERS)
    
    #convert image back to original size and shape
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2
    
def extract_foreground(image):
    """
        This function segements and then finds the contours within an image.
        These contours are bounded by a rectangle, the largest of which
        is used for the Grab Cut algorithm in order to extract the foreground 
        of the image. This extracted foreground can later be used to generate
        a binary mask when used in conjunction with convert_to_bw
        
        Args:
            image (np.ndarray): image from which to extract the foreground
        Returns:
            foreground_extracted (np.ndarray): The extracted foreground from the image      
    """
    img = image.copy()
    
    #kernel for closing edges
    kernel = np.ones((5,5))
    
    #Perform color quantizization
    quantized = quantize_color(img)

    #Threshold the image to segment it. Best results for individual images are
    #labeled as such below
    
    #SWANS
    #ret, threshold = cv2.threshold(quantized, 50, 100, cv2.THRESH_BINARY)
    
    #POLICEMAN
    #ret, threshold = cv2.threshold(quantized, 50, 100, cv2.THRESH_BINARY)
  
    #BIG BEN
    ret, threshold = cv2.threshold(quantized, 100, 125, cv2.THRESH_BINARY)

    #LONDON SCENE
    #ret, threshold = cv2.threshold(quantized, 152, 247, cv2.THRESH_BINARY)

    #De-noise image before edge detection
    blur = cv2.GaussianBlur(threshold, (11,11), 9)


    #Blur edges
    edges = cv2.Canny(blur, 50, 55, 7)
    
    #Close edges to create cohesive edge
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    #Find the external contours of the edge image
    img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_NONE)

    #initial max values for finding largest rectangle in contours
    w_max = 0
    h_max = 0
    
    #iterate through each contour found in the image
    for c in contours:
        #find the bounding rectangles in the contours
        x,y,w,h = cv2.boundingRect(c)
        
        #Identify largest rectangle as foreground component
        if (h >= h_max and w >= w_max):
            r = (x, y, w, h)
            
    #Copy to preserve original
    foreground_extracted = image.copy()
    
    #Create initial mask of zeros and foreground and background of zeros
    mask = np.zeros(foreground_extracted.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    
    #Extract the area bounded by rectangle r and create mask
    cv2.grabCut(foreground_extracted, mask, r, background, foreground, 5, 
                cv2.GC_INIT_WITH_RECT)  
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    foreground_extracted = foreground_extracted * mask2[:,:,np.newaxis]
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)    

    return foreground_extracted


def convert_to_bw(image):
    """ This function converts a grayscale image to black and white.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        numpy.ndarray: The black and white image.
    """
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape
    pixel_val = 0
    
    #iterate through all rows and columns
    for i in range(rows):
        for j in range(cols):
            pixel_val = img[i][j]
            
            #convert any pixel values above 128 to 255, and any below to 0
            if (pixel_val > 1):
                img[i][j] = 255
            else:
                img[i][j] = 0
    
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
    return img


def painting(image, downsample, filter_steps):
    """ Function to take an image and apply a painting filter.
        Applies a bilateral filter with a gaussian pyramid in order to better
        preserve edges. Utilizes OpenCV's pencilSketch function with a sigma s
        of 20, sigma r of 0.09 and shade factor of 0.01. The result of this function 
        is then given a weight of 0.6 and added to a blurred color version of the
        original input image
        
        Args:
            image (numpy.ndarray): image to be "painted"
            downsample (int): the number of levels in Guassian pyramid
            filter_steps (int): the number of steps to take in applying the
                bilateral filter
        Returns:
            painted (numpy.ndarray): image with a painted appearance
    
    """
    
    img = image.copy()
    
    #Gaussian pyramids with bilateral filter to preserve edges
    for i in range(downsample):
        img = cv2.pyrDown(img)
        
    for i in range (filter_steps):
        img = cv2.bilateralFilter(img, d=5, sigmaColor=5, sigmaSpace=3)
        
    for i in range(downsample):
        img = cv2.pyrUp(img)
   
    #Obtain "pencil sketch" from OpenCV
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=20, sigma_r=0.09 , shade_factor=0.01)
    
    #Using dst_gray to BGR yielded a result that had muted color that I
    #preferred, however, for brighter colors, use dst_color with addWeighted
    dst_gray = cv2.cvtColor(dst_gray, cv2.COLOR_GRAY2BGR)

    #blur original to reduce detail
    blurred = image.copy()
    blurred = cv2.blur(img, (3,3))
    blurred = np.uint8(blurred)
    
    #Add the sketch image to the blurred image to combin the two
    painted = cv2.addWeighted(blurred, .4, dst_gray, .6, 0)
         
    return painted 

def painting_gamma(image, downsample, filter_steps, gamma):
    """ Function to take an image and apply a painting filter.
        Applies a bilateral filter with a gaussian pyramid in order to better
        preserve edges. Utilizes OpenCV's pencilSketch function with a sigma s
        of 75, sigma r of 0.09 and shade factor of 0.03. The result of this function 
        is then given a weight of 0.6 and added to a blurred color version of the
        original input image. It creates a table mapping a given gamma
        value to corrected intesity, and performs a look up for each color value
        in the image and adjusts it accordingly
        
        Args:
            image (numpy.ndarray): image to be "painted"
            downsample (int): the number of levels in Guassian pyramid
            filter_steps (int): the number of steps to take in applying the
                bilateral filter
            gamma (double): the gamma value to use for intensity addjustment
        Returns:
            painted (numpy.ndarray): image with a painted appearance
    
    """
    
    img = image.copy()
    
    #Gaussian pyramids with bilateral filter to preserve edges
    for i in range(downsample):
        img = cv2.pyrDown(img)
        
    for i in range (filter_steps):
        img = cv2.bilateralFilter(img, d=5, sigmaColor=5, sigmaSpace=3)
        
    for i in range(downsample):
        img = cv2.pyrUp(img)
    
    #Obtain "pencil sketch" from OpenCV        
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=75, sigma_r=0.09, shade_factor=0.03)            
    dst_gray = cv2.cvtColor(dst_gray, cv2.COLOR_GRAY2BGR)

    #blur image to reduce detail
    blurred = image.copy()
    blurred = cv2.blur(img, (3,3))

    blurred = np.uint8(blurred)
    painted = cv2.addWeighted(blurred, .4, dst_gray, .6, 0)
    
    #perform table lookup using gamma value
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
    painted = cv2.LUT(painted, table)        

    return painted 

def line_draw(image):
    """ Function to take an image and apply a line drawing filter.
        The contours in an input image are found and assigned a thickness of 2.
        An additional background texture image is read in and sized to match
        the input image.
        The contoured image is then blurred and added to the background
        texture image with weights of .35 and .65 respectively.
        
        Args:
            image (numpy.ndarray): image to be "drawn" over
            
        Returns:
            output (numpy.ndarray): image with a drawn appearance
    
    """
    img = image.copy()
    
    #read in background for paper appearance
    paper = cv2.imread("ink-paper.jpg", cv2.IMREAD_COLOR)

    paper = cv2.resize(paper, (img.shape[1], img.shape[0]))

    img = cv2.medianBlur(img, 5)
    edges = cv2.Canny(img, 100 , 125)

    c_img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                                  cv2.CHAIN_APPROX_NONE)
 
    #iterate through each contour found in the image
    for c in contours:
        #draw contours on image. Can vary intensity of lines
        #c_img = cv2.drawContours(c_img, c, -1, (125,125,0), 4)
        c_img = cv2.drawContours(c_img, c, -1, (255,255,255), 2)    
    
    #Invert the line drawing
    c_img = 255 - c_img
    c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)

    c_img_blur = cv2.blur(c_img, (5,5))
     
    #convert to BGR to enable adding
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    edges = np.uint8(edges)  
    c_img_blur = np.uint8(c_img_blur)
    
    #add blurred and contoured to paper to create an overlay/blend
    output = cv2.addWeighted(c_img_blur, .35, paper, .65, 0)
    output = np.uint8(output)
   
    return output 


def sketch(image, downsample, filter_steps):
    """ Function to take an image and apply a sketch filter.
        Applies a bilateral filter with a gaussian pyramid in order to better
        preserve edges. Utilizes OpenCV's pencilSketch function with a sigma s
        of 20, sigma r of 0.09 and shade factor of 0.01 on the filtered image.
        The filtered image is also used with a Canny edge detector, the result of
        which is subsequently used to find the contours. The contoured image is then
        added to the sketch image with respective weights of 0.4 and 0.6.
 
        Args:
            image (numpy.ndarray): image to be "painted"
            downsample (int): the number of levels in Guassian pyramid
            filter_steps (int): the number of steps to take in applying the
                bilateral filter
        Returns:
            output (numpy.ndarray): image with a sketched appearance
    
    """
    img = image.copy()
    
    #Gaussian pyramid with bilateral filter to preserve edges
    for i in range(downsample):
        img = cv2.pyrDown(img)
        
    for i in range (filter_steps):
        img = cv2.bilateralFilter(img, d=5, sigmaColor=5, sigmaSpace=3)
        
    for i in range(downsample):
        img = cv2.pyrUp(img)
 
    dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=20, sigma_r=0.09 , shade_factor=0.01)
    dst_gray = cv2.cvtColor(dst_gray, cv2.COLOR_GRAY2BGR)
    
    edges = cv2.Canny(img, 75 , 100)

    #find the contours
    c_img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_NONE)

    #iterate through each contour found in the image
    for c in contours:

        #draw contours on image
        c_img = cv2.drawContours(c_img, c, -1, (75,75,0), 2)

    #Invert the line drawing
    c_img = 255 - c_img
    
    #convert to BGR to enable adding
    c_img = cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)
    c_img = cv2.blur(c_img, (7,7))

    c_img = np.uint8(c_img)
    
    #add contoured to "sketched" image to create an overlay/blend
    output = cv2.addWeighted(c_img, .4, dst_gray, .6, 0)

    return output 

def main():
    foreground = cv2.imread('big-ben.jpg', cv2.IMREAD_COLOR)
    foreground = cv2.resize(foreground, (800, 600))

    #copy to preserve original image
    foreground_copy = foreground.copy()
    
    #extract foreground to generate a color mask
    color_mask = extract_foreground(foreground_copy)
    
    #convert color mask to binary mask for blending
    mask = convert_to_bw(color_mask)
    mask = cv2.resize(mask, (800, 600))
    
    #blurring the mask yields smoother edges in the composite image
    blur = cv2.GaussianBlur(mask, (11,11), 3)
    mask = np.array(blur)

    foreground = np.array(foreground)
    
    #Choose method to try:
    #illustrated = sketch(foreground_copy, 1, 4)
    #illustrated = painting(foreground_copy, 1, 4)
    #illustrated = painting_gamma(foreground_copy, 1, 4, .3) #last arg is gamma value
    illustrated = line_draw(foreground_copy)
    
    #Create binary mask from bw mask
    mask = mask / 255
    
    #Switch foreground/background to be illustrated
    #illustrated = mask * illustrated
    #foreground_copy = (1.0 - mask) * foreground_copy
    
    #Switch foreground/background to be illustrated
    illustrated = (1.0 - mask) * illustrated
    foreground_copy = mask * foreground_copy
  
    out = foreground_copy + illustrated
    out = out.astype(np.uint8)
    
    #Convert to grayscale if desired
    #out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Composite Image", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
if __name__ == "__main__": main()