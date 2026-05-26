import os
from random import seed, shuffle
import cv2
import keras.backend as K
import segmentation_models as sm
import tensorflow as tf
from PIL import Image
from keras.callbacks import EarlyStopping
from keras_unet_collection._model_swin_unet_2d import swin_unet_2d
from keras_unet_collection._model_transunet_2d import transunet_2d
from numpy import array, zeros, expand_dims, uint8, ndarray
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tqdm.keras import TqdmCallback
import gc

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

IMG_HEIGHT = 1536
IMG_WIDTH = 1536

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


def createSimpleDataset(startIndex, imagesList, masksList, images, masks, transforms):
    for idImage, (filename, fullName) in enumerate(tqdm(imagesList.items())):
        if masksList is not None:
            mask = cv2.imread(masksList[filename])
            mask = mask[:, :, 0]
            mask = cropAndResize(mask, filename)
            mask = expand_dims(mask, axis=-1)
        else:
            mask = None

        image = cv2.imread(fullName)
        image = cropAndResize(image, filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


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


def createDataset(imagesList, masksList, transformsBasic, basicSkipProportion, transformsMore, moreSkipProportion):
    basicLength = len(imagesList)
    basicSize = 0 if basicSkipProportion == 1 else (basicLength if basicSkipProportion == 0 else int(float(basicLength) * (1 - basicSkipProportion)))
    moreSize = 0 if moreSkipProportion == 1 else (int(float(basicLength) * (1 - moreSkipProportion)) if transformsMore is not None and 0 <= moreSkipProportion < 1 else 0)
    images = zeros((basicSize + moreSize, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = uint8)

    if masksList is not None:
        masks = zeros((basicSize + moreSize, IMG_HEIGHT, IMG_WIDTH, 1), dtype = bool)
    else:
        masks = None

    imagesNames = list(imagesList.keys())

    if basicSize > 0:
        shuffle(imagesNames)
        basicImagesList = {filename: fullName for filename, fullName in imagesList.items() if filename in imagesNames[:basicSize]}
        createSimpleDataset(startIndex = 0, imagesList = basicImagesList, masksList = masksList, images = images, masks = masks, transforms = transformsBasic)

    if moreSize > 0:
        shuffle(imagesNames)
        moreImagesList = {filename: fullName for filename, fullName in imagesList.items() if filename in imagesNames[:moreSize] }
        createSimpleDataset(startIndex = basicSize, imagesList = moreImagesList, masksList=masksList, images=images, masks=masks, transforms = transformsMore)

    return images, masks


transformsBasic = None
transformsMore = None


validFilenames = [ filename for filename in os.listdir(masksPath) if filename.endswith("png") and not filename.startswith(".") ]

validTrain, validTest = train_test_split(validFilenames, test_size = 0.1, random_state = SEED)

imagesFilesTrain, maskFilesTrain = getFiles(trainPath, masksPath, validTrain)
imagesFilesTest,  maskFilesTest  = getFiles(trainPath, masksPath, validTest)

if TRAINING_MODE:
    X_train, y_train = createDataset(imagesList = imagesFilesTrain, masksList = maskFilesTrain,
                                     transformsBasic = transformsBasic, basicSkipProportion = 0.3,
                                     transformsMore = transformsMore, moreSkipProportion = 1)
    X_test,  y_test  = createDataset(imagesList = imagesFilesTest,  masksList = maskFilesTest,
                                     transformsBasic = None, basicSkipProportion = 0,
                                     transformsMore = None, moreSkipProportion = 1)


def diceLoss(targetsPure, inputsPure, smooth = 1e-6):
    targets = tf.cast(K.flatten(targetsPure), tf.float32)
    inputs = tf.cast(K.flatten(inputsPure), tf.float32)

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


settings = \
{
    'model': swin_unet_2d((IMG_HEIGHT, IMG_WIDTH, 3), filter_num_begin=64,
                               n_labels=1, num_mlp=1024,
                               depth=4, stack_num_down=2, stack_num_up=2,
                               patch_size=(4, 4), num_heads=[4, 8, 8, 8],
                               window_size=[4, 2, 2, 2],
                               output_activation='Sigmoid', shift_window=True,
                               name='swin_unet'),
    "optimizer":  tf.keras.optimizers.Adam(learning_rate = 0.001),
    'train': TRAINING_MODE,
    "continue-train": False,
    "saved-model": "models-swin/swin.ckpt",
    "batch-size": 1,
    "epochs": 5,
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

    K.clear_session()

else:
    model.load_weights(os.path.join(root, settings["saved-model"])).expect_partial()


imagesCheck, masksEmpty = getFiles(checkPath, None, None)

X_test, empty = createDataset(imagesList = imagesCheck, masksList = None,
                              transformsBasic = None, basicSkipProportion = 0,
                              transformsMore = None, moreSkipProportion = 1)


for imageIndex in tqdm(range(len(imagesCheck))):
    imageName = list(imagesCheck.keys())[imageIndex]

    image = X_test[imageIndex]

    predictedMask = (model.predict(expand_dims(image, axis=0))[0].squeeze() * 255).astype(uint8)

    predictedImage = Image.fromarray(restoreSize(predictedMask, imageName))
    predictedImage = predictedImage.point(lambda pixel: 255 if pixel > 200 else 0)

    predictedImage.save(os.path.join(root, "result/" + imageName))

