from pathlib import Path
import numpy as np

from chainer.dataset.dataset_mixin import DatasetMixin
from chainer.dataset import concat_examples, to_device

def load_dataset(dataset_root_dir):
    pass


def convert_bbox2yolo(size, box):
    """
    [xmin, xmax, ymin, ymax] -> [xcenter, ycenter, width, height] (normalized, 0.0 ~ 1.0)
    """
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
    
def convert_yolo2bbox(size, box):
    """
    [xcenter, ycenter, width, height] -> [xmin, xmax, ymin, ymax] (normalized, 0.0 ~ 1.0)
    """
    w, h = size
    xmin = int(round(w*(box[0] - box[2]/2))) + 1
    xmax = int(round(w*(box[0] + box[2]/2))) + 1
    ymin = int(round(h*(box[1] - box[3]/2))) + 1
    ymax = int(round(h*(box[1] + box[3]/2))) + 1
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
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        image, bbox, label = self.dataset[i]
        h, w, _ = image.shape
        size = (w, h)

        _bbox = []
        for j in range(len(bbox)):
            _bbox.append(convert_bbox2yolo(size, bbox))
        bbox = np.array(_bbox).astype(np.float32)

        image = image.transpose(2, 0, 1)

        return image, bbox, label
