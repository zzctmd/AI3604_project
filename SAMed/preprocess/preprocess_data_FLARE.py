import os
import time
import argparse
from glob import glob

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--src_path', type=str,
                   default='../../dataset/FLARE_origin', help='download path for FLARE data')
parser.add_argument('--dst_path', type=str,
                   default='../../dataset/FLARE/trainset', help='root dir for data')#trainset可更改为testset
parser.add_argument('--use_normalize', action='store_true', default=True,
                   help='use normalize')
args = parser.parse_args()

test_data = [1,3,5,7,9,11,13,15,17,19,31,33,35,37,39,41,43,45,47,49]

#hashmap = {1:5, 2:2, 3:1, 4:8, 5:7, 6:0, 7:0, 8:0, 9:4, 10:0, 11:6, 12:0, 13:3}

hashmap = {6:0,7:0,8:0,10:0,12:0,5:7,11:6,4:8,1:5,9:4,3:1,13:3,2:2}
# 1: spleen
# 2: right kidney
# 3: left kidney
# 4: gallbladder
# 5: liver
# 6: stomach
# 7: aorta
# 8: pancreas

def preprocess_train_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/train_npz", exist_ok=True)

    a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][11:15]

        if int(number) in test_data:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape
        
        image_data = np.transpose(image_data, (2, 1, 0))  # [D, W, H]
        label_data = np.transpose(label_data, (2, 1, 0))

        #counter = 1
        for k in hashmap.keys():
            #assert counter == k
            #counter += 1
            label_data[label_data == k] = hashmap[k]

        for dep in range(D):
            save_path = f"{args.dst_path}/train_npz/case{number}_slice{dep:03d}.npz"
            np.savez(save_path, label=label_data[dep,:,:], image=image_data[dep,:,:])
    pbar.close()


def preprocess_valid_image(image_files: str, label_files: str) -> None:
    os.makedirs(f"{args.dst_path}/test_vol_h5", exist_ok=True)

    a_min, a_max = -125, 275
    b_min, b_max = 0.0, 1.0

    pbar = tqdm(zip(image_files, label_files), total=len(image_files))
    for image_file, label_file in pbar:
        # **/imgXXXX.nii.gz -> parse XXXX
        number = image_file.split('/')[-1][11:15]

        if int(number) not in test_data:
            continue

        image_data = nib.load(image_file).get_fdata()
        label_data = nib.load(label_file).get_fdata()

        image_data = image_data.astype(np.float32)
        label_data = label_data.astype(np.float32)

        image_data = np.clip(image_data, a_min, a_max)
        if args.use_normalize:
            assert a_max != a_min
            image_data = (image_data - a_min) / (a_max - a_min)

        H, W, D = image_data.shape

        image_data = np.transpose(image_data, (2, 1, 0))
        label_data = np.transpose(label_data, (2, 1, 0))

        #counter = 1
        for k in hashmap.keys():
            #assert counter == k
            #counter += 1 
            label_data[label_data == k] = hashmap[k]

        save_path = f"{args.dst_path}/test_vol_h5/case{number}.npy.h5"
        f = h5py.File(save_path, 'w')
        f['image'] = image_data
        f['label'] = label_data
        f.close()
    pbar.close()


if __name__ == "__main__":
    data_root = f"{args.src_path}"

    # String sort
    image_files = sorted(glob(f"{data_root}/images/*.nii"))
    label_files = sorted(glob(f"{data_root}/labels/*.nii/*.nii"))
    
    preprocess_train_image(image_files, label_files)
    preprocess_valid_image(image_files, label_files)