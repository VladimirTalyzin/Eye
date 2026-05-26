import glob
import math
import os
from statistics import mean

import cv2
from numpy import zeros, uint8
from tqdm import tqdm

resultPath = "result"

sourcePath = "result_0.5611"
destinationPath = "result_0.57009"

checkPath = "check"


allContours = []
addContours = []

allContoursSquare = 0
addContoursSquare = 0


def contoursChanged(image, maskSource, maskDestination) -> bool:
    global allContoursSquare
    global addContoursSquare

    contours, hierarchy = cv2.findContours(maskSource[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None or hierarchy is None:
        return maskDestination

    for contour, treeData in zip(contours, hierarchy[0,:,:]):
        area = cv2.contourArea(contour)

        # если это вложенный контур, то пропускаем
        if treeData[3] != -1:
            continue

        allContours.append(contour)

        allContoursSquare += area

        if (550 <= area <= 1050) and checkContourDifference(image, contour):
            addContours.append(contour)
            addContoursSquare += area

            cv2.fillPoly(maskDestination, pts=[contour], color=(255, 255, 255))

    return maskDestination

meanRed   = 105.98
meanGreen = 118.57
meanBlue  = 136.04

threshold = 15


def checkContourDifference(image, contour) -> bool:
    mask = zeros(image.shape, uint8)
    cv2.drawContours(mask, contour, -1, 255, -1)
    (red, green, blue, alpha) = cv2.mean(image, mask=mask[:, :, 0])

    return math.sqrt((meanRed - red) ** 2 + (meanGreen - green) ** 2 + (meanBlue - blue) ** 2) < threshold



for filename in tqdm(glob.glob(sourcePath + "/*.png")):
    pureName = filename.split("\\")[1]

    maskSource = cv2.imread(filename)
    maskDestination = cv2.imread(os.path.join(destinationPath, pureName))
    image = cv2.imread(os.path.join(checkPath, pureName))

    cv2.imwrite(os.path.join(resultPath, pureName), contoursChanged(image, maskSource, maskDestination),  [cv2.IMWRITE_PNG_COMPRESSION, 9])



print("Contours: ", round(100.0 * len(addContours) / len(allContours), 2), "%")
print("Added: ", round(100.0 * addContoursSquare / allContoursSquare, 2), "%")