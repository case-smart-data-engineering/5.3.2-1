# 算法示例

## 使用指南
展示mask_rcnn图片分割效果：
右键test.py文件，并选择‘Run current file in interactive window’
若希望更改测试图片，更改test.py文件中‘img_path’属性（位于第40行）
测试图片位于'./img/original'文件夹下，对应效果图位于'./img/result'文件夹下。

# 文件及程序用途
coco91_indices.json COCO数据集物体类别文件
draw_box_utils.py 目标框绘制
test.py mask_rcnn.py 图片分割演示程序
network_files/boxes.py 目标框位置确定
network_files/det_utils.py 抽样程序
network_files/fast_rcnn_framework.py fast_rcnn框架
network_files/image_list.py 将图片预处理为相同大小
network_files/mask_rcnn.py mask_rcnn网络结构
network_files/roi_head.py roi层
network_files/rpn_function.py rpn
network_files/transform.py 图片标准化处理
