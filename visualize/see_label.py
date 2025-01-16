import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def visualize_label(npz_file, output_folder):
    # 加载npz文件
    data = np.load(npz_file)
    
    # 获取label数据
    label = data['label']
    label= np.transpose(label, (0,1))
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 构建输出文件路径
    output_file = os.path.join(output_folder, 'label_visualization.png')
    
    # 保存label数据为.png文件
    plt.imsave(output_file, label, cmap='gray')
    print(f"Label visualization saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize label data from npz file.')
    parser.add_argument('--npz_file', type=str,default='../dataset/Synapse_12/train_npz/case0002_slice103.npz', help='Path to the npz file.')
    parser.add_argument('--output_folder', type=str,default='./visualize_output/traindata_label', help='Folder to save the output image.')
    
    args = parser.parse_args()
    
    visualize_label(args.npz_file, args.output_folder)