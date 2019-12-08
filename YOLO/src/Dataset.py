from pathlib import Path
import numpy as np
import cv2

from PIL import Image
from pycocotools.coco import COCO

from chainer.dataset.dataset_mixin import DatasetMixin
from chainer.dataset import concat_examples, to_device

from YOLOv3.lib.image import random_detection


def load_dataset(dataset_root_dir=Path(__file__).parent / Path("../dataset/COCO"),
                 json_fname="instances_train2014.json",
                 category_names=["bicycle", "car", "airplane"]):
    ann_file = dataset_root_dir / Path("annotations/{}".format(json_fname)) 
    img_dir = dataset_root_dir / Path(json_fname[json_fname.find("_") + 1:json_fname.find(".")])
    coco = COCO(ann_file)
    imgIds = set()
    catIds = []
    for cat_name in category_names:
        catId = coco.getCatIds(catNms=cat_name)
        catIds.append(catId[0])
        imgIds |= set(coco.getImgIds(catIds=catId))

    images = coco.loadImgs(list(imgIds))
    dataset = []

    for img_id, image in zip(imgIds, images):
        #image = np.asarray(Image.open(img_dir / Path(images[0]['file_name'])))
        image = img_dir / Path(image['file_name'])
        bboxes = []
        labels = []
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
            catId = ann['category_id']
            if catId in catIds:
                labels.append(catIds.index(catId)) # TODO: refactor 毎回indexで検索するのは遅い?->dict
                bboxes.append(ann['bbox'])
        dataset.append([image, bboxes, labels])

    return dataset
    

def convert_bbox2yolo(size, box):
    """
    [xmin, xmax, ymin, ymax] -> [xcenter, ycenter, width, height] (normalized, 0.0 ~ 1.0)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)
    
def convert_yolo2bbox(size, box, normalized=True):
    """
    [xcenter, ycenter, width, height] -> [xmin, xmax, ymin, ymax]
    """
    w, h = size if normalized else 1, 1
    xmin = int(round(w * (box[0] - box[2] / 2))) + 1
    xmax = int(round(w * (box[0] + box[2] / 2))) + 1
    ymin = int(round(h * (box[1] - box[3] / 2))) + 1
    ymax = int(round(h * (box[1] + box[3] / 2))) + 1
    return (xmin, xmax, ymin, ymax)

def concat_yolo(batch, device=None):
    images = []
    bbox = []
    label = []
    for b in batch:
        images.append(b[0])
        bbox.append(to_device(device, np.array(b[1], dtype=np.float32)))
        label.append(to_device(device, np.array(b[2], dtype=np.int32)))
    images = concat_examples(images, device)
    return images, bbox, label
    

class YOLODataset(DatasetMixin):
    """
    YOLO 用のデータセット mixin クラス。
    args:
        dataset (list) : [[image, bbox, label], ...] のリスト
    """
    def __init__(self, dataset, crop_size=(512, 512)):
        self.dataset = dataset
        self.crop_size = crop_size

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        image_path, bbox, label = self.dataset[i]
        image = np.asarray(Image.open(image_path))
        if len(image.shape) == 2:
            h, w, = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            h, w, _ = image.shape
        size = (w, h)

        _bbox = []
        for j in range(len(bbox)):
            _bbox.append(convert_bbox2yolo(size, bbox[j]))
        bbox = np.array(_bbox).astype(np.float32)
        label = np.array(label, np.int32)

        image, bbox, label = random_detection(
            image, bbox, label, self.crop_size, 0)

        image = image.transpose(2, 0, 1)
        image = (image / 255.0).astype(np.float32)

        return image, bbox, label
