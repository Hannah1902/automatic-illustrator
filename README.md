# automatic-illustrator
Filter Function Descriptions and Techniques

sketch():
Apply bilateral filter to levels of Gaussian pyramid in order to preserve edges. Perform contour detection to identify edges. Draw contours with a width of 2, as this value yielded the best appearance of a line drawing when compared to other values that appeared too thick. Blur the contoured image with a kernel of 3x3. The application of the blur gives it an appearance of imperfection and shade, rather than the perfectly precise and defined lines of edge detection. The blur in this function must be slight, as blurring too much decreases the definition too greatly and the lines did not show after the images were added with their respective weights. Attain “sketch” effect from OpenCV’s pencilSketch function. Add the two images, assigning weights to each, where the contoured image has a weight of 0.4 and the image sketch has a weight of 0.6.


painting():
Apply bilateral filter to levels of Gaussian pyramid in order to preserve edges. Attain “sketch” effect from OpenCV’s pencilSketch function. Blur the original image with a kernel of 3x3 in order to reduce harshness and blend details. Add the two images, assigning weights to each, where the blurred image has a weight of 0.4 and the image sketch has a weight of 0.6.


painting_gamma():
Apply bilateral filter to levels of Gaussian pyramid in order to preserve edges. Attain “sketch” effect from OpenCV’s pencilSketch function. Blur the original image with a kernel of 3x3 in order to reduce line harshness and blend details. Add the two images, assigning weights to each, where the contoured image has a weight of 0.4 and the image sketch has a weight of 0.6. Create a table mapping pixel values to corrected gamma output and apply the intensity correction to the original image.


line_draw():
Apply median blur to reduce noise for image blending and find contours. Draw contours with a width of 2, as this value yielded the best appearance of a line drawing when compared to other values that appeared too thick. Blur the contoured image with a kernel of 5x5. The blur gives it an appearance of imperfection and shade, rather than the perfectly precise and defined lines of edge detection. Read in a jpg of a paper texture. Add the two images, assigning weights to each, where the contoured image has a weight of 0.4 and the image sketch has a weight of 0.6
 
 
Automatic Foreground Extraction:
This was performed by first quantizing the color in order to reduce the amount of color variation in an image and then subsequently performing contour detection on the quantized image. From the set of detected contours, the largest was identified as the foreground, and labeled as such and extracted using OpenCV’s built-in grabCut algorithm. This extracted foreground was used to then create a binary mask by using a convert_to_bw function defined in the project. By blending this binary mask with the illustrated image, it was possible to attain a composite image of illustration and reality. This method also allows for an easy swapping of background and foreground, depending on the artistic effect desired. To do this, it is only necessary to switch whether the illustration or original image is to be multiplied by the inverse of the mask, thereby switching which parts of the image are grabbed by which part of the mask.
