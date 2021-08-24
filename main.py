import cv2
import utils


original_image = cv2.imread("cat.jpeg") #change path too absolute path of your image

augmented_image = utils.horizontal_flip(original_image)
cv2.imshow("result", augmented_image)
cv2.imshow("original", original_image)
cv2.waitKey(0)