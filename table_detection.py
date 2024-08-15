# Script to try and detect the table outline on a snooker screen shot

### What do I need to figure out:
# detect table corners myself
# detect when a new shot happens

import cv2
import numpy as np
from matplotlib import pyplot as plt

# reading image
img = cv2.imread("data/images/test_shapes.png")

# converting image into grayscale image
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_green = np.array([45, 100, 20])
upper_green = np.array([75, 255, 255])

mask = cv2.inRange(hsv, lower_green, upper_green)

contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

i = 0
breakpoint()
# list for storing names of shapes
for contour in contours:

    # here we are ignoring first counter because
    # findcontour function detects whole image as shape
    if i == 0:
        i = 1
        continue

    breakpoint()

    # cv2.approxPloyDP() function to approximate the shape
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

    # using drawContours() function
    cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)

    # finding center point of shape
    M = cv2.moments(contour)

    # If it's 0, make it really close it doesn't really matter
    if M["m00"] == 0:
        M["m00"] = 1

    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    # Check to see if we've found 4 sided objects
    # if len(approx) == 4:
    cv2.putText(
        img,
        "Quadrilateral",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

# displaying the image after drawing contours
cv2.imshow("shapes", img)

cv2.waitKey(0)
