import os
import nibabel as nib
import matplotlib.image as mpimg
clock =0 
def load_nii(file_path):
    return nib.load(file_path)

def save_slice_gt(data, slice_index):
    output_folder = './visualize_output/test_output'
    os.makedirs(output_folder, exist_ok=True)
    slice_data = data[:, :, slice_index]
    mpimg.imsave(os.path.join(output_folder, 'test35_gt.png'), slice_data, cmap='gray')

def save_slice_img(data, slice_index):
    output_folder = './visualize_output/test_output'
    os.makedirs(output_folder, exist_ok=True)
    slice_data = data[:, :, slice_index]
    mpimg.imsave(os.path.join(output_folder, 'test35_img.png'), slice_data, cmap='gray')

def save_slice_pred(data, slice_index):
    output_folder = './visualize_output/test_output'
    os.makedirs(output_folder, exist_ok=True)
    slice_data = data[:, :, slice_index]
    mpimg.imsave(os.path.join(output_folder, 'test35_pred.png'), slice_data, cmap='gray')

def process_folder(file_path):
    global clock
    nii = load_nii(file_path)
    data = nii.get_fdata()
    slice_index = data.shape[2] // 2
      # 中间切片
    if clock == 0:
        save_slice_gt(data, slice_index)
        clock = 1
    elif clock == 1:
        save_slice_img(data, slice_index)
        clock = 2
    elif clock == 2:
        save_slice_pred(data, slice_index)
#可将三个nii文件路径放入列表中        
file_paths = ['../output/FLARE/ranksize_1/predictions/case0035_gt.nii.gz',
              '../output/FLARE/ranksize_1/predictions/case0035_img.nii.gz',
              '../output/FLARE/ranksize_1/predictions/case0035_pred.nii.gz'
              ]


for file_path in file_paths:
    process_folder(file_path)