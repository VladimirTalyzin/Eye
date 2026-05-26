import glob
from skimage import measure
from numpy import expand_dims, float32, array
import cv2
from PIL import Image
from skimage.draw import polygon
from tqdm import tqdm

path = "result_1536_final"
preparedPath = "result_prepared"

def prepareContours(mask, threshold, minArea):
    contours = measure.find_contours(mask, threshold)
    for contour in contours:
        # добавляем ещё одну ось к данным контуров, чтобы они соответствовали формату OpenCV
        contourWithAxis = expand_dims(contour.astype(float32), 1)
        # получаем класс UMat OpenCV для контура
        umatContour = cv2.UMat(contourWithAxis)
        # получаем площадь контура
        area = cv2.contourArea(umatContour)
        if area <= minArea:
            rowIndexes, columnIndexes = polygon(contour[:, 0], contour[:, 1])
            mask[rowIndexes, columnIndexes] = 0

    return mask

def prepareFromImage(image):
    # noinspection PyTypeChecker
    imageArray = array(image)
    newImageArray = prepareContours(imageArray, 0.5, 200)
    #newImageArray = prepareContours(imageArray, 0.5, 100)
    return Image.fromarray(newImageArray)

def prepareAndSaveImage(source, destination):
    prepareFromImage(source).save(destination)

def prepareAndSaveImageFile(source, destination):
    prepareFromImage(Image.open(source)).save(destination)
    #exit(0)


for filename in tqdm(glob.glob(path + "/*.png")):
    pureName = filename.split("\\")[1]
    prepareAndSaveImageFile(filename, preparedPath + "/" + pureName)
