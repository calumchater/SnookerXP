import numpy as np
import cv2
import matplotlib.pyplot as plt


def np_to_int_tuple(colour):
    # From format: (np.uint8(22), np.uint8(26), np.uint8(253)) to (22, 26, 253)
    # because it was breaking my code :()

    return tuple(map(int, colour))


def to_hsv(img):
    """
    Convert an image from BGR to HSV colorspace
    """
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)


# Classifies a colour
def classify(rgb_tuple):
    # eg. rgb_tuple = (22,26,253)

    # add as many colors as appropriate here, but for
    # the stated use case you just want to see if your
    # pixel is 'more red' or 'more green'
    colors = {
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
    }

    manhattan = lambda x, y: abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2])
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)

    return color


def create_colour_range(colour, colour_range=45):
    lower_colour = np.zeros(3, dtype="uint8")
    upper_colour = np.zeros(3, dtype="uint8")

    for i in range(0, 3):
        if colour[i] < colour_range:
            lower_colour[i] = 0
        else:
            lower_colour[i] = colour[i] - colour_range

        if colour[i] > (255 - colour_range):
            upper_colour[i] = 255
        else:
            upper_colour[i] = colour[i] + colour_range

    return lower_colour, upper_colour


def get_contours(hsv, lower_color, upper_color, filter_radius=7):
    """
    Returns the contours generated from the given color range
    """
    # Threshold the HSV image to get only cloth colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # use a median filter to get rid of speckle noise
    median = cv2.medianBlur(mask, filter_radius)

    # get the contours of the filtered mask
    # this modifies median in place!
    contours = cv2.findContours(median, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours


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


# Nasty ass function which can probably be improved with some type conversion learning
def count_pixel_colour_frequency(image):
    colors, count = np.unique(
        image.reshape(-1, image.shape[-1]), axis=0, return_counts=True
    )

    # Returns the indexes to sort the prev arrays in descending order
    sorted_colours_indexes = np.argsort(-count)

    return colors[sorted_colours_indexes]


def dist_between(x1, y1, x2, y2):
    """
    Compute the distance between points (x1,y1) and (x2,y2)
    """

    return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def get_UL_coord(contour, pad=10):
    """
    Get the upper left coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(dist_between(c[0][0], c[0][1], 0, 0))

    return (
        contour[get_index_of_min(dists)[0]][0][0] - pad,
        contour[get_index_of_min(dists)[0]][0][1] - pad,
    )


def get_UR_coord(contour, imgXmax, pad=10):
    """
    Get the upper right coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(dist_between(c[0][0], c[0][1], imgXmax, 0))

    return (
        contour[get_index_of_min(dists)[0]][0][0] + pad,
        contour[get_index_of_min(dists)[0]][0][1] - pad,
    )


def get_LL_coord(contour, imgYmax, pad=10):
    """
    Get the lower left coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(dist_between(c[0][0], c[0][1], 0, imgYmax))

    return (
        contour[get_index_of_min(dists)[0]][0][0] - pad,
        contour[get_index_of_min(dists)[0]][0][1] + pad,
    )


def get_LR_coord(contour, imgXmax, imgYmax, pad=10):
    """
    Get the lower right coordinate of the contour.
    """
    dists = []
    for c in contour:
        dists.append(dist_between(c[0][0], c[0][1], imgXmax, imgYmax))

    return (
        contour[get_index_of_min(dists)[0]][0][0] + pad,
        contour[get_index_of_min(dists)[0]][0][1] + pad,
    )


def show_image(img):
    cv2.imshow("lmao", img)
    cv2.waitKey(0)


def get_index_of_min(Data_List):
    """
    Return as list of the indexes of the minmum values in a 1D array of data
    """

    # make sure data is in a standard list, not a numpy array
    if type(Data_List).__module__ == np.__name__:
        Data_List = list(Data_List)

    # return a list of the indexes of the minimum values. Important if there is >1 minimum
    return [i for i, x in enumerate(Data_List) if x == min(Data_List)]


def get_index_of_max(Data_List):
    """
    Return as list of the indexes of the maximum values in a 1D array of data
    """

    # make sure data is in a standard list, not a numpy array
    if type(Data_List).__module__ == np.__name__:
        Data_List = list(Data_List)

    # return a list of the indexes of the max values. Important if there is >1 maximum
    return [i for i, x in enumerate(Data_List) if x == max(Data_List)]
