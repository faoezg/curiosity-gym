import matplotlib.cm
from matplotlib.colors import ListedColormap
import numpy as np

def get_cmap_with_alpha(cmap_name: str):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    colours = cmap(np.linspace(0,1,cmap.N))
    alphas =  np.linspace(0,1,cmap.N)

    colours[:, -1] = alphas

    return ListedColormap(colours, cmap_name + "_alpha")

def one_hot_encode(content_list, labels_list) -> np.ndarray:
    numpy_labels = np.asarray(labels_list)
    label_count = numpy_labels.max() # assuming not 0 index
    return np.eye(label_count)[content_list]

def one_hot_encode_zero_indexed(content_list, labels_list) -> np.ndarray:
    numpy_labels = np.asarray(labels_list)
    label_count = numpy_labels.max() + 1 # assuming 0 index
    return np.eye(label_count)[content_list]

# TODO WHICH ONE?
# def one_hot_encode(size: int, index: int) -> list[int]:
#     one_hot = [0] * size
#     one_hot[index] = 1
#     return one_hot
# 
# def one_hot_encode_zero_indexed(size: int, index: int) -> list[int]:
#     one_hot = [0] * (size + 1)
#     one_hot[index] = 1
#     return one_hot
