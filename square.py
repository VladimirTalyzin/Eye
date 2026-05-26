import glob
from numpy import ndarray, zeros
import cv2
from tqdm import tqdm

left  = ["141.png", "332.png", "455.png", "657.png", "1028.png", "1042.png", "939.png"]
right = ["197.png"]
exclude = ["206.png"]

offsetX = 1624 - 1536
offsetY = 1536 - 1232

def expandImageHeight(image, height):
    half = (height - image.shape[0]) // 2
    image_extended = ndarray((height,) + image.shape[1:], dtype=image.dtype)

    image_extended[half:, :] = 0
    image_extended[half:image.shape[0] + half, :] = image
    image_extended[image.shape[0] + half:, :] = 0

    return image_extended

def cropAndResize(image, filename):
    y = 0
    h = 1232
    w = 1536

    if filename in exclude:
        return image
    elif filename in left:
        x = 0
    elif filename in right:
        x = offsetX
    else:
        x = offsetX // 2

    crop = image[y:y + h, x:x + w]

    extended = expandImageHeight(crop, 1536)

    return extended


def restoreSize(image, filename):
    black = zeros((1232, 1624), dtype="uint8")

    if filename in exclude:
        return image
    if filename in left:
        start = 0
    elif filename in right:
        start = offsetX
    else:
        start = offsetX // 2

    x = 0
    y = offsetY // 2
    h = 1232
    w = 1536

    crop = image[y:y + h, x:x + w]

    black[0:h, start:(w + start)] = crop

    return black


def test(path):
    for filename in tqdm(glob.glob(path + "/*.png")):
        image = cv2.imread(filename)

        square = cv2.countNonZero(image[:, :, 0])

        name = filename.replace("\\", "/").split("/")[1]

        crop = cropAndResize(image[:, :, 0], name)
        squareCrop = cv2.countNonZero(crop)

        if square != squareCrop:
            print(filename, square - squareCrop)

        restored = restoreSize(crop, name)

        cv2.imwrite("result/" + name, restored)


#test("masks-2")
test("result_prepared_0.5673")