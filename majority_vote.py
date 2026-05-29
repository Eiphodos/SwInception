import argparse
import glob
import os

import nibabel as nib
import numpy as np


def load_nifti_data(files_all_folds, curr_index, n_folds):
    pred_data = []
    for f in range(n_folds):
        fn = files_all_folds[f][curr_index]
        pred_data.append(nib.load(fn))
    return tuple(pred_data)

def get_nifti_fdata(nifti_data, n_folds):
    fdata = []
    for f in range(n_folds):
        fdata.append(nifti_data[f].get_fdata())
    return tuple(fdata)

def get_class_votes(fdata, n_folds, n_classes):
    vol_shape = fdata[0].shape
    fold_votes = np.zeros((n_folds, n_classes, vol_shape[0], vol_shape[1], vol_shape[2]), dtype=np.uint8)
    for f in range(n_folds):
        fold_vote = np.zeros((n_classes, vol_shape[0], vol_shape[1], vol_shape[2]), dtype=np.uint8)
        for c in range(1, n_classes):
            fold_vote[c] += (fdata[f] == c).astype(np.uint8)
        fold_votes[f] = fold_vote
    class_votes = np.sum(fold_votes, axis=0)
    class_votes[0,:,:] = class_votes[0,:,:] + 1
    return class_votes

def get_new_label(fdata, n_folds, n_classes):
    class_votes = get_class_votes(fdata, n_folds, n_classes)
    return np.argmax(class_votes, axis=0)

parser = argparse.ArgumentParser()
parser.add_argument('--in_folder', type=str)
parser.add_argument('--n_classes', type=int)
parser.add_argument('--subfolder_prefix', type=str, default='rs')
parser.add_argument('--folds', type=int, default=5)

args = parser.parse_args()

root_folder = args.in_folder
pred_prefix = args.subfolder_prefix

out_folder = os.path.join(root_folder, 'voted_output', pred_prefix)
os.makedirs(out_folder, exist_ok=True)



pred_files_all_folds = []
pred_rs_files_all_folds = []

folds = args.folds
n_classes = args.n_classes

for fold in range(folds):
    fold_folder = os.path.join(root_folder, 'Fold' +  str(fold))

    pred_files = glob.glob(os.path.join(fold_folder, pred_prefix, '*'))

    pred_files_all_folds.append(pred_files)

    print(len(pred_files))

n_files = len(pred_files)

for i in range(n_files):
    pred_data = load_nifti_data(pred_files_all_folds, i, folds)

    pred_fdata = get_nifti_fdata(pred_data, folds)

    new_pred = get_new_label(pred_fdata, folds, n_classes)

    _, filename = os.path.split(pred_files_all_folds[0][i])

    affine = pred_data[0].affine

    nib.save(nib.Nifti1Image(new_pred.astype(np.uint8), affine),
                     os.path.join(out_folder, filename))
    



    