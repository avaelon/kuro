import cv2
import utils

#change the image to the corresponding directory accordingly
original_image = cv2.imread("/train/animals/cats/cat.jpeg") 

augmented_image = utils.horizontal_flip(original_image)
cv2.imshow("result", augmented_image)
cv2.imshow("original", original_image)
cv2.waitKey(0)
