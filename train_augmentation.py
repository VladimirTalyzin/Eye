import os
from random import seed, shuffle
import cv2
import keras.backend as K
import segmentation_models as sm
import tensorflow as tf
from PIL import Image
from albumentations import Compose, ShiftScaleRotate, ImageCompression, HueSaturationValue, HorizontalFlip, VerticalFlip, \
    Blur, ISONoise, OneOf, MotionBlur, Sharpen, GaussNoise, RandomGamma, RandomBrightnessContrast
from keras.callbacks import EarlyStopping
from numpy import array, zeros, expand_dims, uint8
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

IMG_HEIGHT = 1024
IMG_WIDTH = 1024

IMG_CHANNELS = 3

SEED = 500

TRAINING_MODE = False

sm.set_framework('tf.keras')
tf.keras.utils.set_random_seed(SEED)
seed(SEED)

root = ''

masksPath = os.path.join(root, 'masks-2-pure')
trainPath = os.path.join(root, 'train')
checkPath = os.path.join(root, 'check')


def getFiles(trainData, resultData, validNames):
    if resultData is not None:
        masks = { filename: os.path.join(resultData, filename) for filename in os.listdir(resultData) if filename.endswith(".png") and (validNames is None or filename in validNames) }
    else:
        masks = None

    images = { filename: os.path.join(trainData, filename) for filename in os.listdir(trainData) if filename.endswith(".png") and (validNames is None or filename in validNames) }

    return images, masks


def createSimpleDataset(startIndex, imagesList, masksList, images, masks, transforms):
    for idImage, (filename, fullName) in enumerate(tqdm(imagesList.items())):
        if masksList is not None:
            mask = cv2.imread(masksList[filename])
            mask = cv2.resize(mask, (IMG_HEIGHT, IMG_WIDTH))
            mask = mask[:, :, 0]
            mask = expand_dims(mask, axis=-1)
        else:
            mask = None

        image = cv2.imread(fullName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))

        if transforms is not None:
            augmented = transforms(image = array(image), mask = mask)
            del image
            image = augmented["image"]
            if mask is not None:
                del mask
                mask = augmented["mask"]

        if mask is not None:
            masks[startIndex + idImage] = mask

        images[startIndex + idImage] = image


def createDataset(imagesList, masksList, transformsBasic, transformsMore, moreSkipProportion = 0.0):
    dataLength = len(imagesList)
    moreSize = int(float(dataLength) * (1.0 - moreSkipProportion)) if transformsMore is not None and 0 <= moreSkipProportion < 1 else 0
    images = zeros((dataLength + moreSize, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = uint8)

    if masksList is not None:
        masks = zeros((dataLength + moreSize, IMG_HEIGHT, IMG_WIDTH, 1), dtype = bool)
    else:
        masks = None

    createSimpleDataset(startIndex = 0, imagesList = imagesList, masksList = masksList, images = images, masks = masks, transforms = transformsBasic)

    if moreSize > 0:
        imagesNames = list(imagesList.keys())
        shuffle(imagesNames)
        moreImagesList = {filename: fullName for filename, fullName in imagesList.items() if filename in imagesNames[:moreSize] }
        createSimpleDataset(startIndex = dataLength, imagesList = moreImagesList, masksList=masksList, images=images, masks=masks, transforms = transformsMore)

    return images, masks


transformsBasic = Compose(
[
    ShiftScaleRotate(scale_limit = 0.1, rotate_limit = 30, p = 0.3),
    VerticalFlip(p = 0.1),
    HorizontalFlip(p = 0.3),
    Blur(blur_limit = 1, p = 0.08)
])

transformsMore = Compose(
[
    ShiftScaleRotate(scale_limit = 0.2, rotate_limit = 60, p = 0.6),
    ISONoise(p = 0.7),
    ImageCompression(quality_lower = 85, quality_upper = 100, p = 0.5),
    HueSaturationValue(hue_shift_limit = 15, sat_shift_limit = 15, val_shift_limit = 15, p = 0.5),
    RandomBrightnessContrast(),
    VerticalFlip(p = 0.3),
    HorizontalFlip(p = 0.8),
    GaussNoise(p = 0.1),
    RandomGamma(p = 0.1),

    OneOf(
    [
        Blur(blur_limit=2, p = 0.2),
        MotionBlur(blur_limit = 4, p = 0.1),
        Sharpen(p = 0.2)
    ], p = 0.5)
])

validFilenames = [ filename for filename in os.listdir(masksPath) if filename.endswith("png") and not filename.startswith(".") ]

validTrain, validTest = train_test_split(validFilenames, test_size = 0.1, random_state = SEED)

imagesFilesTrain, maskFilesTrain = getFiles(trainPath, masksPath, validTrain)
imagesFilesTest,  maskFilesTest  = getFiles(trainPath, masksPath, validTest)

if TRAINING_MODE:
    X_train, y_train = createDataset(imagesList = imagesFilesTrain, masksList = maskFilesTrain, transformsBasic = transformsBasic, transformsMore = transformsMore, moreSkipProportion = 0.3)
    X_test,  y_test  = createDataset(imagesList = imagesFilesTest,  masksList = maskFilesTest,  transformsBasic = None, transformsMore = None)


def diceLoss(targetsPure, inputsPure, smooth = 1e-6):
    targets = K.flatten(targetsPure)
    inputs = K.flatten(inputsPure)
    targets = tf.cast(targets, tf.float32)
    inputs = tf.cast(inputs, tf.float32)

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


settings = \
{
    'model': sm.Unet('efficientnetb0', classes = 1, activation = 'sigmoid'),
    "optimizer": "Adam",
    'train': TRAINING_MODE,
    "continue-train": False,
    "saved-model": "models-augmentation/unet-efficientnetb0.ckpt",
    "batch-size": 6,
    "epochs": 2,
    "callbacks": [EarlyStopping(monitor = 'val_iou_score', mode = 'max', patience = 10, verbose = 0, restore_best_weights = True)]
}

model = settings['model']

if settings['train']:
    if settings["continue-train"]:
        model.load_weights(os.path.join(root, settings["saved-model"])).expect_partial()

    model.compile(settings["optimizer"], loss = diceLoss, metrics=[sm.metrics.iou_score])

    gc.collect()

    # noinspection PyUnboundLocalVariable
    model.fit(x=X_train,y=y_train, batch_size=settings["batch-size"], epochs=settings["epochs"], validation_data = (X_test, y_test), verbose = 0,
              shuffle = True, callbacks = settings["callbacks"] + [TqdmCallback(verbose=2)])
    model.save_weights(os.path.join(root, settings["saved-model"]))

    del X_train
    del y_train

else:
    model.load_weights(os.path.join(root, settings["saved-model"])).expect_partial()

K.clear_session()
imagesCheck, masksEmpty = getFiles(checkPath, None, None)

X_test, empty = createDataset(imagesList = imagesCheck, masksList = None, transformsBasic = None, transformsMore = None)


for imageIndex in tqdm(range(len(imagesCheck))):
    imageName = list(imagesCheck.keys())[imageIndex]

    image = X_test[imageIndex]

    predictedMask = (model.predict(expand_dims(X_test[imageIndex], axis=0))[0].squeeze() * 255).astype(uint8)

    predictedImage = Image.fromarray(predictedMask)
    predictedImage = predictedImage.resize((1624, 1232), Image.LANCZOS).point(lambda pixel: 255 if pixel > 220 else 0)

    predictedImage.save(os.path.join(root, "result/" + imageName))

