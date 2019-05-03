import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def plot_cam(attr, xi, cmap='jet', alpha=0.5):
    attr -= attr.min()
    attr /= (attr.max() + 1e-20)
    
    #print attr.shape
    plt.imshow(xi)	
    plt.imshow(attr,cmap="jet", alpha=0.5)
    #plt.imshow(a, alpha=alpha, cmap=cmap)
    

def plot_bbox(bboxes,xi, linewidth=2):
    ax = plt.gca()
    ax.imshow(xi)

    if not isinstance(bboxes[0], list):
        bboxes = [bboxes]

    for bbox in bboxes:
        #print bbox
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] , bbox[3],
                                 linewidth=linewidth, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

	
    ax.axis('off')
    return ax

