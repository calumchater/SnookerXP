import cv2
import numpy as np
import matplotlib.pyplot as plt


def Test():

    img = cv2.imread("data/images/table1.png")

    hsv = to_hsv(img)

    max_color = 

    classify(tuple(max_color))    

    # lower_color, upper_color = get_cloth_colour(hsv)

    # contours = get_contours(hsv, lower_color, upper_color, 7)

    # TableContour = mask_table_bed(contours)

    # warp = transform_to_overhead(img, TableContour)

    # # Now the table is cropped and warped, lets find the balls
    # # hsv = ToHSV(warp)

    # lower_color, upper_color = get_cloth_color(hsv)

    # contours = GetContours(hsv, lower_color, upper_color, 17)

    # # BallData = FindTheBalls(warp, contours)
    # # print(lower_color)
    # # ShowCueBall(BallData)



def ShowCueBall(BallData):

    data = BallData[1][2]

    # this mask does not reflect the boundary between data and nodata.
    mask = cv2.inRange(data, (0, 0, 10), (180, 255, 255))

    #    cv2.imshow('result1',mask)
    #    cv2.imshow('result',data)
    #
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()

    hist = cv2.calcHist([data], [0], mask, [180], [0, 180])

    plt.plot(hist)
    plt.show()

    hist = cv2.calcHist([data], [1], mask, [256], [0, 256])

    plt.plot(hist)
    plt.show()

    hist = cv2.calcHist([data], [2], mask, [256], [0, 256])

    plt.plot(hist)
    plt.show()





def get_cloth_colour(hsv, search_width=45):
    """
    Find the most common HSV values in the image.
    In a well lit image, this will be the cloth
    """

    hist = cv2.calcHist([hsv], [1], None, [180], [0, 180])
    h_max = get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    s_max = get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    v_max = get_index_of_max(hist)[0]

    # define range of green color in HSV
    lower_color = np.array(
        [h_max - search_width, s_max - search_width, v_max - search_width]
    )
    upper_color = np.array(
        [h_max + search_width, s_max + search_width, v_max + search_width]
    )

    return lower_color, upper_color


def mask_table_bed(contours):
    """
    Mask out the table bed, assuming that it will be the biggest contour.
    """

    # The largest area should be the table bed
    areas = []
    breakpoint()
    for c in contours:
        areas.append(cv2.contourArea(c))

    # return the contour that delineates the table bed
    largest_contour = get_index_of_max(areas)
    return contours[largest_contour[0]]






def transform_to_overhead(img, contour):
    """
    Get the corner coordinates of the table bed by finding the minumum
    distance to the corners of the image for each point in the contour.

    Transform code is built upon code from: http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    """

    # get dimensions of image
    height, width, channels = img.shape

    # find the 4 corners of the table bed
    UL = get_UL_coord(contour)
    UR = get_UR_coord(contour, width)
    LL = get_LL_coord(contour, height)
    LR = get_LR_coord(contour, width, height)

    # store the coordinates in a numpy array
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = [UL[0], UL[1]]
    rect[1] = [UR[0], UR[1]]
    rect[2] = [LR[0], LR[1]]
    rect[3] = [LL[0], LL[1]]

    # get the width at the bottom and top of the image
    widthA = dist_between(LL[0], LL[1], LR[0], LR[1])
    widthB = dist_between(UL[0], UL[1], UR[0], UR[1])

    # choose the maximum width
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = maxWidth * 2  # pool tables are twice as long as they are wide

    # construct our destination points which will be used to
    # map the image to a top-down, "birds eye" view
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return warp





def find_the_balls(img, contours, similarity_threshold=5):
    """
    Find and circle all of the balls on the table.

    Currently struggles with balls on the rail. Not yet tested on clusters.

    Returns a three-tuple containing a tuple of x,y coords, a radius and the masked
    out area of the image. Needs to be made into a ball object.
    """

    # dimensions of image
    height, width, channels = img.shape

    # compare the difference in area of a min bounding circle and the cotour area
    diffs = []
    indexes = []

    for i, contour in enumerate(contours):
        contourArea = cv2.contourArea(contour)
        (x, y), radius = cv2.minEnclosingCircle(contour)

        circleArea = 3.141 * (radius**2)
        diffs.append(abs(circleArea - contourArea))
        indexes.append(i)

    sorted_data = sorted(zip(diffs, indexes))

    diffs = [x[0] for x in sorted_data]
    indexes = [x[1] for x in sorted_data]

    # list of center coords as tuples
    centers = []
    radii = []
    masks = []
    for i, d in zip(indexes, diffs):
        # if the contour is a similar shape to the circle it is likely to be a ball.
        if d < diffs[0] * similarity_threshold:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])

            center = (int(x), int(y))
            radius = int(radius)
            # remove .copy() to display a circle round each ball
            cv2.circle(img.copy(), center, radius, (0, 0, 255), 2)
            centers.append(center)
            radii.append(radius)

            circle_img = np.zeros((height, width), np.uint8)
            cv2.circle(circle_img, center, radius, 1, thickness=-1)
            masked_data = cv2.bitwise_and(img, img, mask=circle_img)
            masks.append(masked_data)

    return zip(centers, radii, masks)




Test()
