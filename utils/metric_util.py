# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    print(f'hist.sum(1) is {hist.sum(1)}')
    print(f'hist.sum(0) is {hist.sum(0)}')
    print(f'np.diag(hist) is {np.diag(hist)}')
    if np.isnan(hist.sum(1) + hist.sum(0) - np.diag(hist)).any(): print(f'Denominator has NaN')
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    if np.isnan(output).any(): print(f'NaN in output in fast_hist_crop!')
    if np.isnan(target).any(): print(f'NaN in target in fast_hist_crop!')
    if np.isnan(hist).any(): print(f'NaN in hist in fast_hist_crop!')

    return hist
