import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from natsort import natsorted
import pickle

def get_single_test(test_path):
    img_list = os.listdir(test_path)
    sz = len(img_list)
    test = np.zeros(sz)
    cnt = 0
    for f in natsorted(img_list):
        img = cv2.imread(os.path.join(test_path, f), 0)
        if np.max(img) == 255: test[cnt] = 1
        cnt = cnt + 1
    return test

def gen_sequence(test_array):
    sz = test_array.shape[0] - 10
    sequences = []
    # apply the sliding window technique to get the sequences
    for i in range(sz):
        clip = np.zeros(10)
        for j in range(10):
            clip[j] = test_array[i + j]
        sequences.append(clip)
    return sequences

def get_ground_truth(sequences):
    sz = len(sequences)
    res = np.zeros(sz)
    for seq_num in range(sz):
        if np.any(sequences[seq_num]): res[seq_num] = 1
    return res

if __name__ == '__main__':
    ann_paths = ['E:\\AirSim Recordings\\validation_sets\\sparse_final\\ann_sparse_val', 'E:\\AirSim Recordings\\validation_sets\\medium_final\\ann_medium_val', 'E:\\AirSim Recordings\\validation_sets\\foliage_final\\ann_foliage_val']
    pred_files = ['validation\\val_sparse_new_ssim_deeper_acacia_foliage_rgb.pkl', 'validation\\val_medium_new_ssim_deeper_acacia_foliage_rgb.pkl', 'validation\\val_foliage_new_ssim_deeper_acacia_foliage_rgb.pkl']

    truth  = np.zeros(1)
    for ann in ann_paths:
        test_array = get_single_test(ann)
        seq = gen_sequence(test_array)
        truth = np.append(truth, get_ground_truth(seq))
    truth = truth[1:]

    y_preds = np.zeros(1)
    for pred_file in pred_files:
        with open(pred_file, 'rb') as f:
            pred = pickle.load(f)
            y_preds = np.append(y_preds, pred)
    y_preds = y_preds[1:]

    fpr, tpr, thresh = roc_curve(truth, y_preds)
    auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label = 'AUC = {}'.format(auc))
    plt.legend()
    plt.show()