import cv2
import numpy as np

image = cv2.imread("images/document.jpg")

cv2.imshow("Original", image)

cv2.waitKey(0)
cv2.destroyAllWindows()