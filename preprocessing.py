import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
import umap
from sklearn.preprocessing import MinMaxScaler
import cv2
import seaborn as sns
import torch
from skimage.util import view_as_windows

def Peak2Mat(data,peak,trainPeak,m, n):

    returnMatrix = np.zeros((len(trainPeak),1, m, n))
    for i in range(len(trainPeak)):
        index = np.where(peak == trainPeak[i])
        tomatrix = data[:,index].reshape(m, n)
        returnMatrix[i][0] = tomatrix

    return returnMatrix

def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()  # 归一化
    return cdf

def hist_match(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    source_nonzero = source[source != 0]
    template_nonzero = template[source != 0]

    s_values, bin_idx, s_counts = np.unique(source_nonzero, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template_nonzero, return_counts=True)

    s_cdf = calculate_cdf(s_counts)
    t_cdf = calculate_cdf(t_counts)

    interp_t_values = np.interp(s_cdf, t_cdf, t_values)

    matched = np.zeros_like(source)
    matched[source != 0] = interp_t_values[bin_idx]
    return matched.reshape(oldshape)

def Thist_match(data,m,n):
    data_count = np.where(data != 0, 1, 0)
    data_count = np.sum(data_count, axis = 0)
    target_data = data[:,np.argmax(data_count)]
    returnMatrix = np.zeros_like(data)
    for i in range(len(data[0])):
        if i != np.argmax(data_count):
            matched_data = hist_match(data[:,i].reshape(m, n), target_data.reshape(m, n))
            returnMatrix[:,i] = matched_data.reshape(1,-1)[0]
        else:
            returnMatrix[:, i] = target_data.reshape(1,-1)[0]
    return returnMatrix


def nor_std(data):
    data_sum = np.sum(data, axis=1).reshape(-1, 1)
    data_sum = np.where(data_sum == 0, 1, data_sum)
    data_TIC = data / data_sum

    b = np.percentile(data_TIC, 99.9, axis=0)
    return_data = np.zeros_like(data_TIC)

    for i in range(len(data_TIC[0])):
        da = data_TIC[:, i]
        return_data[:, i] = np.where(da > b[i], b[i], da)

    # return_data = StandardScaler().fit_transform(return_data)
    return_data = MinMaxScaler().fit_transform(return_data)
    return return_data

def extract_patches_with_overlap(imageMat, patch_size, step):
    patch_size = [patch_size,patch_size]
    num1,_,m,n = imageMat.shape
    returnpatches = []

    for num in range(num1):
        image = imageMat[num,0]
        patches = view_as_windows(image, patch_size, step)

        extra_patches = []
        if image.shape[0] % step != 0:
            for i in range(0, image.shape[1] - patch_size[1] + 1, step):
                extra_patches.append(image[-patch_size[0]:, i:i + patch_size[1]])

        if image.shape[1] % step != 0:
            for i in range(0, image.shape[0] - patch_size[0] + 1, step):
                extra_patches.append(image[i:i + patch_size[0], -patch_size[1]:])

        if image.shape[0] % step != 0 and image.shape[1] % step != 0:
            extra_patches.append(image[-patch_size[0]:, -patch_size[1]:])

        if extra_patches:
            extra_patches = np.array(extra_patches)
            patches = np.concatenate((patches.reshape(-1, *patch_size), extra_patches), axis=0)
        returnpatches.append(list(patches))

    returnpatches = np.array(returnpatches)
    a,patchNUMPer,c,d = returnpatches.shape

    return returnpatches.reshape(a*patchNUMPer,1,c,d), patchNUMPer

def reconstruct_image_from_patches(ALLpatches, image_shape, patchNUMPer, patch_size, step):
    returnmat = []
    patch_size = [patch_size,patch_size]
    num = len(ALLpatches) // patchNUMPer
    for i in range(num):
        patches = ALLpatches[i*patchNUMPer:(i+1)*patchNUMPer,0]
        reconstructed_image = np.zeros(image_shape, dtype=patches.dtype)
        overlap_count = np.zeros(image_shape, dtype=np.int32)

        patch_idx = 0
        for i in range(0, image_shape[0] - patch_size[0] + 1, step):
            for j in range(0, image_shape[1] - patch_size[1] + 1, step):
                reconstructed_image[i:i + patch_size[0], j:j + patch_size[1]] += patches[patch_idx]
                overlap_count[i:i + patch_size[0], j:j + patch_size[1]] += 1
                patch_idx += 1

        if image_shape[0] % step != 0:
            for j in range(0, image_shape[1] - patch_size[1] + 1, step):
                reconstructed_image[-patch_size[0]:, j:j + patch_size[1]] += patches[patch_idx]
                overlap_count[-patch_size[0]:, j:j + patch_size[1]] += 1
                patch_idx += 1

        if image_shape[1] % step != 0:
            for i in range(0, image_shape[0] - patch_size[0] + 1, step):
                reconstructed_image[i:i + patch_size[0], -patch_size[1]:] += patches[patch_idx]
                overlap_count[i:i + patch_size[0], -patch_size[1]:] += 1
                patch_idx += 1

        if image_shape[0] % step != 0 and image_shape[1] % step != 0:
            reconstructed_image[-patch_size[0]:, -patch_size[1]:] += patches[patch_idx]
            overlap_count[-patch_size[0]:, -patch_size[1]:] += 1

        reconstructed_image = reconstructed_image / np.maximum(overlap_count, 1)
        returnmat.append(list(reconstructed_image))
    returnmat = np.array(returnmat)
    return returnmat