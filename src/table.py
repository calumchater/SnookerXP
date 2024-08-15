import numpy as np
import cv2
import helpers


img = cv2.imread("data/images/table1.png")

hsv = helpers.to_hsv(img)

colour_distribution = helpers.count_pixel_colour_frequency(hsv)

cloth_colour = []


for colour in colour_distribution:

    if helpers.classify(helpers.np_to_int_tuple(colour)) == "green":
        cloth_colour = colour
        break

lower_colour, upper_colour = helpers.create_colour_range(cloth_colour)

# We now have the probable cloth colour, and we can continue finding the contour
contours = helpers.get_contours(hsv, lower_colour, upper_colour)
# We should have one big contour now

table_contour = helpers.mask_table_bed(contours)


warp = helpers.transform_to_overhead(img, table_contour)


table_center = cv2.moments(table_contour)

# i = 0
# for contour in contours:

#     # here we are ignoring first counter because
#     # findcontour function detects whole image as shape
#     # if i == 0:
#     #     i = 1
#     #     continue

#     breakpoint()

#     # cv2.approxPloyDP() function to approximate the shape
#     # approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

#     # using drawContours() function
#     # cv2.drawContours(img, [contour], -1, (0, 0, 255), 5)

#     # finding center point of shape
#     M = cv2.moments(contour)

#     # # If it's 0, make it really close it doesn't really matter
#     # if M["m00"] == 0:
#     #     M["m00"] = 1

#     # x = int(M["m10"] / M["m00"])
#     # y = int(M["m01"] / M["m00"])

#     # # Check to see if we've found 4 sided objects
#     # cv2.putText(
#     #     img,
#     #     "Quadrilateral",
#     #     (x, y),
#     #     cv2.FONT_HERSHEY_SIMPLEX,
#     #     0.6,
#     #     (255, 255, 255),
#     #     2,
#     # )

# displaying the image after drawing contours
cv2.imshow("shapes", img)
