import numpy as np
import cv2
import json
import glob
from PIL import Image
from tqdm import tqdm


def parse_polygon(coordinates, image_size):
    mask = np.zeros(image_size, dtype=np.float32)

    if len(coordinates) == 1:
        points = [np.int32(coordinates)]
        cv2.fillPoly(mask, points, 1)
    else:
        points = [np.int32([coordinates[0]])]
        cv2.fillPoly(mask, points, 1)

        for polygon in coordinates[1:]:
            points = [np.int32([polygon])]
            cv2.fillPoly(mask, points, 0)

    return mask


def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
    mask = np.zeros(image_size, dtype=np.float32)
    coordinates = shape['coordinates']
    if shape['type'] == 'MultiPolygon':
        for polygon in coordinates:
            mask += parse_polygon(polygon, image_size)
    else:
        mask += parse_polygon(coordinates, image_size)

    return mask


def generateMask(path, image_size) -> np.ndarray:
    with open(path, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
        json_contents = json.load(f)

    mask_channels = np.zeros(image_size, dtype=np.float32)

    if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
        features = json_contents['features']
    elif type(json_contents) == list:
        features = json_contents
    else:
        features = [json_contents]

    for shape in features:
        mask = parse_mask(shape['geometry'], image_size)
        mask_channels = np.maximum(mask_channels, mask)

    return mask_channels * 255


def createMask(imageName, geoJSONName, maskName):
    image = Image.open(imageName)
    mask = generateMask(geoJSONName, (image.height, image.width))
    maskImage = Image.fromarray(mask.astype(np.uint8))
    maskImage.save(maskName)


def createEmptyMask(imageName, maskName):
    image = Image.open(imageName)
    maskImage = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
    maskImage.save(maskName)


collectFiles = {}
for file in glob.glob("train/*.png"):
    idFile = file.split("\\")[1].split(".")[0]
    collectFiles[idFile] = {"png": file}

for file in glob.glob("train/*.geojson"):
    idFile = file.split("\\")[1].split(".")[0]
    if idFile in collectFiles:
        collectFiles[idFile]["geojson"] = file


collectFiles = dict((key, value) for key, value in collectFiles.items() if "geojson" in value)


for idFile, files in tqdm(collectFiles.items()):
    createMask(files["png"], files["geojson"], "masks-2/" + idFile + ".png")


#for file in glob.glob("check/*.png"):
#    idFile = file.split("\\")[1].split(".")[0]
#    createEmptyMask(file, "check-masks/" + idFile + ".png")
