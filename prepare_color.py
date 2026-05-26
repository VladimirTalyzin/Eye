import glob
import math
import os
from statistics import mean

import cv2
from numpy import zeros, uint8
from tqdm import tqdm

resultPath = "result"
sourcePath = "result_prepared_0.5673"
#sourcePath = "result_1536_0.5629"
checkPath = "check"

prepareMode = False


allContours = []
removeContours = []

allContoursSquare = 0
removeContoursSquare = 0
addContoursSquare = 0


# TODO: раздели делить средние цвета по группам контуров по площади

def contoursChanged(image, mask) -> bool:
    global allContoursSquare
    global removeContoursSquare
    global addContoursSquare

    contours, hierarchy = cv2.findContours(mask[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or hierarchy is None:
        return mask

    for contour, treeData in zip(contours, hierarchy[0,:,:]):
        area = cv2.contourArea(contour)

        # если это вложенный контур, 4-й элемент массива информации будет отличен от -1
        if treeData[3] != -1:
            if not prepareMode and area < 60:
                addContoursSquare += area

                cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
            continue

        if prepareMode:
            collectMeanColor(image, contour)

        else:
            allContours.append(contour)

            allContoursSquare += area

            if area < 40 or (area < 245 and checkContourDifference(image, contour)):
                removeContours.append(contour)
                removeContoursSquare += area

                cv2.fillPoly(mask, pts=[contour], color=(0, 0, 0))

    return mask

meanRed   = 105.98
meanGreen = 118.57
meanBlue  = 136.04

threshold = 70

redCollect = []
greenCollect = []
blueCollect = []

def collectMeanColor(image, contour):
    mask = zeros(image.shape, uint8)
    cv2.drawContours(mask, contour, -1, 255, -1)
    (red, green, blue, alpha) = cv2.mean(image, mask=mask[:,:,0])
    redCollect.append(red)
    greenCollect.append(green)
    blueCollect.append(blue)


def checkContourDifference(image, contour) -> bool:
    mask = zeros(image.shape, uint8)
    cv2.drawContours(mask, contour, -1, 255, -1)
    (red, green, blue, alpha) = cv2.mean(image, mask=mask[:, :, 0])

    return math.sqrt((meanRed - red) ** 2 + (meanGreen - green) ** 2 + (meanBlue - blue) ** 2) > threshold



for filename in tqdm(glob.glob(sourcePath + "/*.png")):
    pureName = filename.split("\\")[1]

    mask = cv2.imread(filename)
    image = cv2.imread(os.path.join(checkPath, pureName))

    cv2.imwrite(os.path.join(resultPath, pureName), contoursChanged(image, mask),  [cv2.IMWRITE_PNG_COMPRESSION, 9])


if prepareMode:
    print("Red: ", mean(redCollect), "Green:", mean(greenCollect), "Blue:", mean(blueCollect))

else:
    print("Removed: ", round(100.0 * len(removeContours) / len(allContours), 2), "%")
    print("Added: ", round(100.0 * addContoursSquare / allContoursSquare, 2), "%")
    print("Square: ", round(100.0 * removeContoursSquare / allContoursSquare, 2), "%")