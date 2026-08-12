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


# Contour 
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)

# Find the document

document_contour = None

for contour in contours:
    perimeter = cv2.arcLength(contour, True)

    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approximation) == 4:
        document_contour = approximation
        break


result = image.copy()

cv2.drawContours(
    result,
    [document_contour],
    -1,
    (0, 255, 0),
    3
)

cv2.imshow("Document", result)

cv2.waitKey(0)
cv2.destroyAllWindows()


# Order the four points

def order_points(points):
    points = points.reshape(4, 2)

    ordered = np.zeros((4, 2), dtype=np.float32)

    s = points.sum(axis=1)

    ordered[0] = points[np.argmin(s)]   # top-left
    ordered[2] = points[np.argmax(s)]   # bottom-right

    diff = np.diff(points, axis=1)

    ordered[1] = points[np.argmin(diff)]  # top-right
    ordered[3] = points[np.argmax(diff)]  # bottom-left

    return ordered

points = order_points(document_contour)

# Output 
width = 600
height = 800

destination = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(
    points,
    destination
)

scanned = cv2.warpPerspective(
    image,
    matrix,
    (width, height)
)

cv2.imshow("Scanned Document", scanned)

cv2.waitKey(0)
cv2.destroyAllWindows()