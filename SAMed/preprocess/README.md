# Data preprocessing
This README file records the operation of data preprocessing, corresponding to the operations in `preprocess_data_FLARE.py` and `preprocess_data_Synapse12.py`
1. Please download the Synapse dataset and FLARE dataset from [交大云盘](https://jbox.sjtu.edu.cn/l/q1hgrW) clip the value range to \[-125,275\], normalize the 3D image to \[0-1\], extract the 2D slices from the 3D image for training and store the 3D images as h5 files for inference.

2.According to the setting of Synapse comparision experiment in [SAM-Adapter](https://github.com/tianrun-chen/SAM-Adapter-PyTorch),the map of the semantic labels between the original data and the processed data is shown below.

Organ | Label of the original data | Label of the processed data
------------ | -------------|----
spleen | 1 | 1
right kidney| 2 | 2
left kidney| 3 | 3
gallbladder| 4 | 4
esophagus| 5 | 5
liver| 6 | 6
stomach| 7 | 7
aorta| 8 | 8
inferior vena cava| 9 | 9
inferior vena cava| 10 | 10
pancreas| 11 | 11
adrenal gland| 12,13 |12

3. According to the [data description](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) of the Synapse dataset,and we change the label of FLARE dataset in the same way, the map of the semantic labels between the original data and the processed data is shown below.

Organ | Label of the original data | Label of the processed data
------------ | -------------|----
spleen | 3 | 1
right kidney | 2 | 2
left kidney | 13 | 3
gallbladder | 9 | 4
liver | 1 | 5
stomach | 11 | 6
aorta | 5 | 7
pancreas | 4 | 8

4.值得注意的是这两个代码需要分别运行两遍，一次执行训练集的处理，一次执行测试集的处理，两次间的切换需要更改一些参数。

