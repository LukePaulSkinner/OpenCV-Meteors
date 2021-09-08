# --------------------------Program Summary-------------------------- #
# This is a program used to detect calculate the distance of an object
# The program should be run from command line
# Inputs should be in the form of
# number of images
# Template of left image
# Template of right image
# Using this information and information from the brief
# The program will then output:
# The objects and their distances in each frame followed by
# Any objects identified to be a UFO
# Note: if an object is only visible in one image then it will not be output
# -----------------------End Of Program Summary----------------------- #

# ------------------------Example Input/Output------------------------ #
# As stated in the summary the program takes 3 inputs
# number of images
# Template of left image
# Template of right image
# The user must have these images saved in the same location
# as the program for it to function
# See below for example input
# python3 main.py 50 left-%3.3d.png right-%3.3d.png
# This input takes 50 images named:
# left-000.png to left-049.png and right-000.png to right-049.png
# Below you can see a snippet of an example output of this program
#      Frame   Identity Distance
# .................................................
#         45        Red 2.62E+07
#         45     Yellow 3.11E+07
#         45       Blue 3.13E+07
#         45      Green 2.37E+07
#         45     Orange 2.37E+07
#         45      White 2.84E+07
#         45       Cyan 2.39E+07
#         46        Red 2.40E+07
#         46     Yellow 2.90E+07
# .................................................
# UFO: Cyan
# ---------------------End Of Example Input/Output--------------------- #

# --------------------------Program Structure-------------------Lines- #
# The First set of code in the program is the python imports     76-80
#
# The Second set of code contains the information
# used for running the program e.g. focal-length, distance,
# color ranges and initializing empty arrays                     82-117
#
# The third set of code is the functions used in the program.
# These are used to calculate the centre of an asteroid,
# calculate the distance and check if an asteroid is a UFO       120-188
#
# The fourth set of code is used to check
# that the program was called correctly                          191-201
#
# The fifth set of the code prints the headers and enters the loop
# using the provided arguments                                   203-208
#
# After entering the loop the program will first check that
# the files obtained from the template are valid                 210-225
#
# After confirming the files are valid the program will then
# get the centre of each asteroid in the left and right frame
# using the function "find_centre" and then use this data to
# to call "calculatedist" to get the distance.                   227-264
#
# After getting the centres and distances of each asteroid
# the program will check that the asteroid is visible in
# each image and if it is print it and add it to the array.      266-300
#
# After all valid entries are input into the array I then
# call the "isUFO" function to determine if the object
# is moving in a straight line, if so I add it to the UFO output.
# Finally the program then prints the UFO's                      303-327
# -----------------------End Of Program Structure----------------------- #
# Imports
import sys
import numpy as np
import cv2
from numpy.linalg import norm

# Information from assignment brief
focal = 12
distance = 3500
micronspacing = 10
micronspacingmetre = 0.00001

# Range for blue
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
# Range for red
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
# Range for orange
lower_orange = np.array([20, 100, 100])
upper_orange = np.array([25, 255, 255])
# Range for white
lower_white = np.array([0, 0, 100])
upper_white = np.array([0, 0, 255])
# Range for green
lower_green = np.array([36, 0, 0])
upper_green = np.array([86, 255, 255])
# Range for yellow
lower_yellow = np.array([25, 100, 100])
upper_yellow = np.array([30, 255, 255])
# Range For cyan
lower_cyan = np.array([80, 255, 143])
upper_cyan = np.array([100, 255, 223])

# Initializing empty arrays to store coordinates
yellow = []
orange = []
green = []
blue = []
red = []
white = []
cyan = []


# Find distance based on the center point of two features
# using the distance between cameras and the focal distance
# Returns -1 if not possible
# In this function is using metres as its unit of measurement
def calculatedist(focal, distance, leftfeature, rightfeature):
    if leftfeature == "E" or rightfeature == "E":
        return -1
    differenceX = leftfeature[0] - rightfeature[0]
    # Converting pixels to metres
    differenceX = differenceX * micronspacingmetre
    if differenceX == 0:
        return focal * distance
    else:
        return (focal * distance) / differenceX


# Find the center of a asteroid based on upper and lower HSV values and image
# Works by creating a mask based on upper and lower hsv values.
# It then uses this mask in order to isolate the asteroid.
# From there it finds the centre of the asteroid.
# Then uses the centre coordinates and the with and height of the image
# to find the relative position of the centre.
# If no asteroid can be found then the function returns E
# Returns a list containing x and y coordinate
def find_centre(lower, upper, img):
    img = cv2.imread(img)
    # Getting height and width from image
    y = img.shape[0]
    x = img.shape[1]
    # Converting img to HSV and applying mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    M = cv2.moments(mask)
    # Getting centre of centroid
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        # returning E if none found
        return "E"

    # Transforming to 3d coords
    cX = (cX - x / 2)
    cY = (cY - y / 2)

    return [cX, cY]


# Takes an array of 2d coordinates
# Returns true if asteroid
# Works by creating a line between start and end points
# And then calculates each points distance to the line
# I was unable to implement 3d detection so-
# - if a UFO were to dodge along the z axis it would not be detected
def isUFO(coordinates):
    p1 = np.asarray(coordinates[0])
    p2 = np.asarray(coordinates[-1])
    for i in range(0, len(coordinates)):
        p3 = np.asarray(coordinates[i])
        # Line below modified from
        # https://stackoverflow.com/
        # questions/39840030/distance-between-point-and-a-line-from-two-points
        # This is responsible for calculating distance from expected path
        out = np.abs(np.cross(p2 - p1, p1 - p3)) / norm(p2 - p1)
        # Using 3 to allow for margin of error when getting centre
        if out > 3:
            return True

    return False


# Error if not run from command line/not enough arguments
# Will abort the program if triggered
if len(sys.argv) < 2:
    sys.stderr.write("Incorrect systen arguments,"
                     " Program shold be ran from command line"
                     " with the following arguments:\n")

    sys.stderr.write("the number of frames to be processed\n")
    sys.stderr.write("template for generating filename of left-hand frame\n")
    sys.stderr.write("template for generating filename of right-hand frame\n")
    sys.exit(1)

print('%10s' % "Frame", '%10s' % "Identity", "Distance")
# The Following 4 lines were taken from the assignment brief
nframes = int(sys.argv[1])
for frame in range(0, nframes):
    fn_left = "Images\\"+sys.argv[2] % frame
    fn_right = "Images\\"+sys.argv[3] % frame

    # Code to check that images are reachable
    # Will output the image name that could not be found
    try:
        left_test = open(fn_left)
    except IOError:
        sys.stderr.write("ERROR: " + fn_left + " not accessible")
        sys.exit(1)
    finally:
        left_test.close()
    try:
        right_test = open(fn_right)
    except IOError:
        sys.stderr.write("ERROR: " + fn_right + " not accessible")
        sys.exit(1)
    finally:
        right_test.close()

    # Below I am running my functions to get:
    # The centre point for each asteroid in each frame
    # The distance from the asteroid

    # Distance for red
    lredcentre = find_centre(lower_red, upper_red, fn_left)
    rredcentre = find_centre(lower_red, upper_red, fn_right)
    reddist = calculatedist(focal, distance, lredcentre, rredcentre)

    # Distance for yellow
    lyellowcentre = find_centre(lower_yellow, upper_yellow, fn_left)
    ryellowcentre = find_centre(lower_yellow, upper_yellow, fn_right)
    yellowdist = calculatedist(focal, distance, lyellowcentre, ryellowcentre)

    # Distance for blue
    lbluecentre = find_centre(lower_blue, upper_blue, fn_left)
    rbluecentre = find_centre(lower_blue, upper_blue, fn_right)
    bluedist = calculatedist(focal, distance, lbluecentre, rbluecentre)

    # Distance for green
    lgreencentre = find_centre(lower_green, upper_green, fn_left)
    rgreencentre = find_centre(lower_green, upper_green, fn_right)
    greendist = calculatedist(focal, distance, lgreencentre, rgreencentre)

    # Distance for orange
    lorangecentre = find_centre(lower_orange, upper_orange, fn_left)
    rorangecentre = find_centre(lower_orange, upper_orange, fn_right)
    orangedist = calculatedist(focal, distance, lorangecentre, rorangecentre)

    # Distance for white
    lwhitecentre = find_centre(lower_white, upper_white, fn_left)
    rwhitecentre = find_centre(lower_white, upper_white, fn_right)
    whitedist = calculatedist(focal, distance, lwhitecentre, rwhitecentre)

    # Distance for cyan
    lcyancentre = find_centre(lower_cyan, upper_cyan, fn_left)
    rcyancentre = find_centre(lower_cyan, upper_cyan, fn_right)
    cyandist = calculatedist(focal, distance, lcyancentre, rcyancentre)

    # Adding centres to arrays and printing the distance
    # Red
    if lredcentre != "E" and rredcentre != "E":
        red.append(lredcentre)
        print('%10s' % frame, '%10s' % "Red", '%.2E' % reddist)

    # Yellow
    if lyellowcentre != "E" and ryellowcentre != "E":
        yellow.append(lyellowcentre)
        print('%10s' % frame, '%10s' % "Yellow", '%.2E' % yellowdist)

    # Blue
    if lbluecentre != "E" and rbluecentre != "E":
        blue.append(lbluecentre)
        print('%10s' % frame, '%10s' % "Blue", '%.2E' % bluedist)

    # Green
    if lgreencentre != "E" and rgreencentre != "E":
        green.append(lgreencentre)
        print('%10s' % frame, '%10s' % "Green", '%.2E' % greendist)

    # Orange
    if lorangecentre != "E" and rorangecentre != "E":
        orange.append(lorangecentre)
        print('%10s' % frame, '%10s' % "Orange", '%.2E' % orangedist)

    # White
    if lwhitecentre != "E" and rwhitecentre != "E":
        white.append(lwhitecentre)
        print('%10s' % frame, '%10s' % "White", '%.2E' % whitedist)

    # Cyan
    if lcyancentre != "E" and rcyancentre != "E":
        cyan.append(lcyancentre)
        print('%10s' % frame, '%10s' % "Cyan", '%.2E' % cyandist)


# Calling solve function and if true adding ufo to output
UFOS = "UFO:"

if isUFO(red):
    UFOS += " Red"

if isUFO(yellow):
    UFOS += " Yellow"

if isUFO(blue):
    UFOS += " Blue"

if isUFO(green):
    UFOS += " Green"

if isUFO(orange):
    UFOS += " Orange"

if isUFO(white):
    UFOS += " White"

if isUFO(cyan):
    UFOS += " Cyan"

print(UFOS)
