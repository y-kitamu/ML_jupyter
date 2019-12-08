import matplotlib.pyplot as plt
import matplotlib.patches as patches

from Dataset import convert_yolo2bbox


def plot_image_and_bbox(ax, image, yolo_boxes, labels, is_box_normalized=True):
    ax.imshow(image)

    for yolo, label in zip(yolo_boxes, labels):
        bbox = convert_yolo2bbox(yolo, is_box_normalized)
        width = bbox[1] - bbox[0]
        height = bbox[3] - bbox[2]
        ax.add_patch(patches.Rectangle(
            xy=(bbox[0], bbox[2]), width=width, height=height, fill=False, linewidth=3.0))

    return ax

