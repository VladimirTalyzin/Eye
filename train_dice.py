import os

import cv2
import keras.backend as K
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from PIL import Image
from keras.callbacks import EarlyStopping

from tqdm import tqdm
from tqdm.keras import TqdmCallback

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

sm.set_framework('tf.keras')
sm.framework()

root = ''

def Data_sorting(input_data, target_data):
    if target_data is not None:
        masks = { filename: os.path.join(target_data, filename) for filename in os.listdir(target_data) if filename.endswith("png") and not filename.startswith(".") }
    else:
        masks = None

    images = sorted([os.path.join(input_data, filename) for filename in os.listdir(input_data) if filename.endswith("png") and (target_data is None or filename in masks)])

    if target_data is not None:
        masks = sorted(masks.values())

    return images, masks


input_data_drive_train = os.path.join(root, 'train')
target_data_drive_train = os.path.join(root, 'masks-2-pure')
images_drive_train, masks_drive_train = Data_sorting(input_data_drive_train, target_data_drive_train)

###############################################################################################################

def Create_Dataset(folder_path,is_mask,img_height,img_width,img_channels):
    result = np.zeros((len(folder_path), img_height, img_width, 1 if is_mask else img_channels), dtype=bool if is_mask else np.uint8)

    for idImage, fileName in enumerate(tqdm(folder_path)):
        if not is_mask:
            img = cv2.imread(fileName)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(img_height,img_width))
            result[idImage] = img
        else:
            img = cv2.imread(fileName)
            img = cv2.resize(img, (img_height, img_width))
            img = img[:, :, 0]
            img = np.expand_dims(img, axis=-1)
            result[idImage] = img

    return result

IMG_HEIGHT = 1024
IMG_WIDTH = 1024
IMG_CHANNELS = 3

X_train = Create_Dataset(folder_path=images_drive_train,is_mask=False,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,img_channels=IMG_CHANNELS)
y_train = Create_Dataset(folder_path=masks_drive_train,is_mask=True,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,img_channels=1)


def DiceLoss(targetsPure, inputsPure, smooth = 1e-6):
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
    'train': False,
    "continue-train": False,
    "saved-model": "models-dice/unet-efficientnetb0.ckpt",
    "batch-size": 1,
    "epochs": 30,
    "callbacks": [EarlyStopping(monitor='val_iou_score', mode='max', patience = 8, verbose = 0, restore_best_weights = True)]
}

model = settings['model']

if settings['train']:
    if settings["continue-train"]:
        model.load_weights(os.path.join(root, settings["saved-model"]))

    model.compile(settings["optimizer"], loss=DiceLoss, metrics=[sm.metrics.iou_score])

    model.fit(x=X_train,y=y_train, batch_size=settings["batch-size"],  epochs=settings["epochs"], validation_split = 0.15, verbose = 0,
              callbacks = settings["callbacks"] + [TqdmCallback(verbose=2)])
    model.save_weights(os.path.join(root, settings["saved-model"]))
else:
    model.load_weights(os.path.join(root, settings["saved-model"]))


input_data_drive_test = os.path.join(root, 'check')
images_drive_test, masks_drive_test = Data_sorting(input_data_drive_test, None)

X_test  = Create_Dataset(folder_path=images_drive_test,is_mask=False,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,img_channels=IMG_CHANNELS)


for imageIndex in tqdm(range(len(images_drive_test))):
    imageName = images_drive_test[imageIndex]

    image = X_test[imageIndex]

    predictedMask = (model.predict(np.expand_dims(X_test[imageIndex], axis=0))[0].squeeze() * 255).astype(np.uint8)

    predictedImage = Image.fromarray(predictedMask)
    predictedImage = predictedImage.resize((1624, 1232), Image.LANCZOS).point(lambda pixel: 255 if pixel > 200 else 0)

    predictedImage.save(os.path.join(root, "result/" + imageName.replace("\\", "/").split("/")[-1]))

