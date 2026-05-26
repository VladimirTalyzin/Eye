import glob
import os
import warnings
from datetime import datetime

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

import json

class EyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, transform = None):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.transform = transform
        self._image_files = glob.glob(f"{data_folder}/*.png")

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image / 255, dtype=np.float32)
        return image

    @staticmethod
    def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.float32)
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            for polygon in coordinates:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 1)
        return mask

    @staticmethod
    def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
        """
        Метод для парсинга фигур из geojson файла
        """
        mask = np.zeros(image_size, dtype=np.float32)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size)

        return mask

    def read_layout(self, path: str, image_size: tuple) -> np.ndarray | None:
        """
        Метод для чтения geojson разметки и перевода в numpy маску
        """
        if os.path.exists(path):
            with open(path, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
                json_contents = json.load(f)

            num_channels = 1 + max(self.class_ids.values())
            mask_channels = [np.zeros(image_size, dtype=np.float32) for _ in range(num_channels)]
            #mask = np.zeros(image_size, dtype=np.float32)

            if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
                features = json_contents['features']
            elif type(json_contents) == list:
                features = json_contents
            else:
                features = [json_contents]

            for shape in features:
                channel_id = self.class_ids["vessel"]
                mask = self.parse_mask(shape['geometry'], image_size)
                mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)

            mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)

            return np.stack(mask_channels, axis=-1)
        else:
            return None

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self._image_files[idx]

        # Получаем соответствующий файл разметки
        json_path = image_path.replace("png", "geojson")

        image = self.read_image(image_path)

        mask = self.read_layout(json_path, image.shape[:2])

        if mask is not None:
            sample = {'image': image, 'mask': mask}
        else:
            sample = {'image': image}

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self._image_files)

    # Метод для проверки состояния датасета
    def make_report(self):
        reports = []
        if not self.data_folder:
            reports.append("Путь к датасету не указан")
        if len(self._image_files) == 0:
            reports.append("Изображения для распознавания не найдены")
        else:
            reports.append(f"Найдено {len(self._image_files)} изображений")
        cnt_images_without_masks = sum([1 - len(glob.glob(filepath.replace("png", "geojson"))) for filepath in self._image_files])
        if cnt_images_without_masks > 0:
            reports.append(f"Найдено {cnt_images_without_masks} изображений без разметки")
        else:
            reports.append(f"Для всех изображений есть файл разметки")
        return reports


class DatasetPart(Dataset):
    """
    Обертка над классом датасета для его разбиения на части
    """
    def __init__(self, dataset: Dataset,
                 indices: np.ndarray | None,
                 transform: A.Compose = None):
        self.dataset = dataset
        self.indices = indices

        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        if self.indices is None:
            sample = self.dataset[idx]
        else:
            sample = self.dataset[self.indices[idx]]

        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self) -> int:
        if self.indices is None:
            return len(self)
        else:
            return len(self.indices)


size = 1024
train_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
              A.PadIfNeeded(size, size),
              ToTensorV2(transpose_mask=True),
              ]
eval_list = [A.LongestMaxSize(size, interpolation=cv2.INTER_CUBIC),
              A.PadIfNeeded(size, size),
              ToTensorV2(transpose_mask=True)]

transforms = {'train': A.Compose(train_list), 'test': A.Compose(eval_list)}


dataset = EyeDataset("train")

train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.25)

print(f"Разбиение на train/test : {len(train_indices)}/{len(test_indices)}")

train_dataset = DatasetPart(dataset, train_indices, transform=transforms['train'])
valid_dataset = DatasetPart(dataset, test_indices, transform=transforms['test'])


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                   num_workers=0,
                                   pin_memory=True,
                                   shuffle=True, drop_last=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1,
                                   num_workers=0,
                                   pin_memory=True,
                                   shuffle=True, drop_last=True)


def plot_history(train_history, val_history, title='loss'):
    plt.figure()
    plt.title('{}'.format(title))
    plt.plot(train_history, label='train', zorder=1)

    #points = np.array(val_history)
    steps = list(range(0, len(train_history) + 1, int(len(train_history) / len(val_history))))[1:]

    plt.scatter(steps, val_history, marker='+', s=180, c='orange', label='val', zorder=2)
    plt.xlabel('train steps')

    plt.legend(loc='best')
    plt.grid()

    plt.savefig(title + ".png")


from typing import Tuple, List, Callable, Iterator, Optional, Dict, Any
from collections import defaultdict

class UnetTrainer:
    """
    Класс, реализующий обучение модели
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 device: str, metric_functions: List[Tuple[str, Callable]] = [],
                 epoch_number: int = 0,
                 lr_scheduler: Optional[Any] = None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.device = device

        self.metric_functions = metric_functions

        self.epoch_number = epoch_number

    @torch.no_grad()
    def evaluate_batch(self, val_iterator: Iterator, eval_on_n_batches: int) -> Optional[Dict[str, float]]:
        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(eval_on_n_batches):
            try:
                batch = next(val_iterator)

                xs = batch['image'].to(self.device)
                ys_true = batch['mask'].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break
            ys_pred = self.model.eval()(xs)
            loss = self.criterion(ys_pred, ys_true)

            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        return metrics

    @torch.no_grad()
    def evaluate(self, val_loader, eval_on_n_batches: int = 1) -> Dict[str, float]:
        """
        Вычисление метрик для эпохи
        """
        metrics_sum = defaultdict(float)
        num_batches = 0

        val_iterator = iter(val_loader)

        while True:
            batch_metrics = self.evaluate_batch(val_iterator, eval_on_n_batches)

            if batch_metrics is None:
                break

            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]

            num_batches += 1

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit_batch(self, train_iterator: Iterator, update_every_n_batches: int) -> Optional[Dict[str, float]]:
        """
        Тренировка модели на одном батче
        """

        self.optimizer.zero_grad()

        predictions = []
        targets = []

        losses = []

        for real_batch_number in range(update_every_n_batches):
            try:
                batch = next(train_iterator)

                xs = batch['image'].to(self.device)
                ys_true = batch['mask'].to(self.device)
            except StopIteration:
                if real_batch_number == 0:
                    return None
                else:
                    break

            ys_pred = self.model.train()(xs)
            loss = self.criterion(ys_pred, ys_true)

            (loss / update_every_n_batches).backward()

            losses.append(loss.item())

            predictions.append(ys_pred.cpu())
            targets.append(ys_true.cpu())

        self.optimizer.step()

        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)

        metrics = {'loss': np.mean(losses)}

        for metric_name, metric_fn in self.metric_functions:
            metrics[metric_name] = metric_fn(predictions, targets).item()

        return metrics

    def fit_epoch(self, train_loader, update_every_n_batches: int = 1) -> Dict[str, float]:
        """
        Одна эпоха тренировки модели
        """

        metrics_sum = defaultdict(float)
        num_batches = 0

        train_iterator = iter(train_loader)

        while True:
            batch_metrics = self.fit_batch(train_iterator, update_every_n_batches)

            if batch_metrics is None:
                break

            for metric_name in batch_metrics:
                metrics_sum[metric_name] += batch_metrics[metric_name]

            num_batches += 1

        metrics = {}

        for metric_name in metrics_sum:
            metrics[metric_name] = metrics_sum[metric_name] / num_batches

        return metrics

    def fit(self, train_loader, num_epochs: int,
            val_loader = None, update_every_n_batches: int = 1,
            ) -> Dict[str, np.ndarray]:
        """
        Метод, тренирующий модель и вычисляющий метрики для каждой эпохи
        """

        summary = defaultdict(list)

        def save_metrics(metrics: Dict[str, float], postfix: str = '') -> None:
          # Сохранение метрик в summary
            nonlocal summary, self

            for metric in metrics:
                metric_name, metric_value = f'{metric}{postfix}', metrics[metric]

                summary[metric_name].append(metric_value)

        for _ in tqdm(range(num_epochs - self.epoch_number), initial=self.epoch_number, total=num_epochs):
            self.epoch_number += 1

            train_metrics = self.fit_epoch(train_loader, update_every_n_batches)

            with torch.no_grad():
                save_metrics(train_metrics, postfix='_train')

                if val_loader is not None:
                    test_metrics = self.evaluate(val_loader)
                    save_metrics(test_metrics, postfix='_test')

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        summary = {metric: np.array(summary[metric]) for metric in summary}

        return summary



# F1-мера
class SoftDice:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def __call__(self, predictions: List[Dict[str, torch.Tensor]],
                 targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        numerator = torch.sum(2 * predictions * targets)
        denominator = torch.sum(predictions + targets)
        return numerator / (denominator + self.epsilon)

# Метрика полноты
class Recall:
    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b

    def __call__(self, predictions: List[Dict[str, torch.Tensor]],
                 targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        numerator = torch.sum(predictions * targets)
        denominator = torch.sum(targets)

        return numerator / (denominator + self.epsilon)

# Метрика точности
class Accuracy:
    def __init__(self, epsilon=1e-8, b=1):
        self.epsilon = epsilon
        self.a = b*b

    def __call__(self, predictions: list, targets: list) -> torch.Tensor:
        numerator = torch.sum(targets)
        denominator = torch.sum(predictions * targets)

        return numerator / (denominator + self.epsilon)

def make_metrics():
    soft_dice = SoftDice()
    recall = Recall()
    accuracy = Accuracy()

    def exp_dice(pred, target):
        return soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])

    def exp_accuracy(pred, target):
        return accuracy(torch.exp(pred[:, 1:]), target[:, 1:])

    def exp_recall(pred, target):
        return recall(torch.exp(pred[:, 1:]), target[:, 1:])

    return [('exp_dice', exp_dice),
            ('accuracy', exp_accuracy),
            ('recall',   exp_recall),
            ]


training = True

if training:
    torch.cuda.empty_cache()
    # Подгружаем модель и задаём функцию потерь
    model = smp.Unet('resnet50', activation='sigmoid', classes=1).cuda()

    def make_criterion():
        soft_dice = SoftDice()

        def exp_dice(pred, target):
            return 1 - soft_dice(torch.exp(pred[:, 1:]), target[:, 1:])

        return exp_dice

    criterion = make_criterion()


    optimizer = torch.optim.Adam(model.parameters(), 0.0001)


    # Обучаем модель
    trainer = UnetTrainer(model, optimizer, criterion, 'cuda', metric_functions=make_metrics())
    summary = trainer.fit(train_loader, num_epochs=20, val_loader=valid_loader)

    startTimeString = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    torch.save(model, "eye_" + startTimeString + ".pth")

    plot_history(summary['loss_train'], summary['loss_test'])
    plot_history(summary['accuracy_train'], summary['accuracy_test'], "accuracy")
    plot_history(summary['recall_train'], summary['recall_test'], "recall")

else:
    model = torch.load("eye_2022-09-11-23-16-22.pth")

checkDataset = EyeDataset("check")
checkImages = DatasetPart(checkDataset, None, transform=transforms['test'])

for i, sample in enumerate(checkImages):
    image = sample["image"].to("cuda")

    predictedMask = model.predict(image.unsqueeze(0)).squeeze().cpu().numpy()
    predictedMask[predictedMask < 0.8] = 0
    predictedMask[predictedMask >= 0.8] = 255
    #predict = model.predict(image.unsqueeze(0)).squeeze().cpu().numpy()
    #mask = predict
    #prediction = model.predict(image.unsqueeze(dim=0))
    #mask = (255 if torch.exp(prediction[0]) > 0.5 else 0)

    #prediction = model.eval()()

    #image = (image.cpu() * 255).type(torch.uint8)
    #pred_ask = (torch.exp(prediction[0]) > 0.5).cpu()

    #image_with_mask = draw_segmentation_masks(image, pred_ask)
    #image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)

    cv2.imwrite("result/" + str(i) + ".png", predictedMask)


    #image_with_mask = draw_segmentation_masks(image, true_mask.type(torch.bool))
    #cv2.imwrite("result/" + str(test_indices[i]) + "_2.png", image_with_mask)
    #image_with_mask = np.moveaxis(image_with_mask.cpu().numpy(), 0, -1)
    #cv2.imwrite("result/" + str(test_indices[i]) + "_3.png", image_with_mask)