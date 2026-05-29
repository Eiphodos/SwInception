import argparse
import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', type=str, help='Folder that contains the data')
parser.add_argument('--output_name', type=str, help='Filename to save json as')
args = parser.parse_args()

metadata_dict = {
    "name": "Some name",
    "description": "Some description",
    "reference": "Some reference",
    "licence":"-",
    "release":"-",
    "tensorImageSize": "3D",
    "modality": {
        "0": "CT"
    }
}

mode = "Train"
keys = ("image", "label")
data_folder = os.path.join(args.input_folder, mode)
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
images = ['./' + mode + '/' + os.path.split(f)[1] for f in images]
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
labels = ['./' + mode + '/' + os.path.split(f)[1] for f in labels]
n_files = len(images)
print("Number of files: {} for mode {}".format(n_files, "training"))
train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images, labels)]
print("Number of files: {} for mode {}".format(len(train_files), "training"))
metadata_dict['numTraining'] = n_files
metadata_dict['training'] = train_files

mode = "Validation"
key = "image"
data_folder = os.path.join(args.input_folder, mode)
images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
images = ['./' + mode + '/' + os.path.split(f)[1] for f in images]
n_files = len(images)
print("Number of files: {} for mode {}".format(n_files, "test"))
test_files = [{key: img} for img in images]
print("Number of files: {} for mode {}".format(len(test_files), "test"))
metadata_dict['numTest'] = n_files
metadata_dict['test'] = test_files

output_path = os.path.join(args.input_folder, args.output_name)

with open(output_path, "w") as wf:
    json.dump(metadata_dict, wf, indent=4)
