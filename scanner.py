import cv2
import numpy as np

# Read document
image = cv2.imread("images/document.jpg")

cv2.imshow("Original", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Convert document into gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 75, 200)

cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

