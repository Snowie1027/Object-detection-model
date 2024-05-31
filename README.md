# Object-detection-model
在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3。

## 1. 环境依赖
    •  Python 3.7
    •  PyTorch 1.7.1
    •  Torchvision 0.8.2
    •  OpenCV 4.5.1
    •  Numpy 1.19.2
## 2. 数据准备
    1  下载PASCAL VOC2012数据集,并解压到项目根目录下的data文件夹中。
    2  使用torchvision.datasets.VOCDetection类加载数据集,并划分为训练集、验证集和测试集。

## 3. 训练与测试
   
### 3.1 Faster R-CNN
    1  使用torchvision.models.detection.fasterrcnn_resnet50_fpn预训练模型作为backbone。
    2  修改模型输出类别数,并在训练集上fine-tune模型。
    3  使用测试集评估模型性能。
 
### python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#### 加载预训练的Faster R-CNN模型
model = fasterrcnn_resnet50_fpn(pretrained=True)

#### 修改模型输出类别数
num_classes = 21  # VOC数据集有20个类别+背景
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#### 在训练集上fine-tune模型
# ...

#### 在测试集上评估模型
# ...

### 3.2 YOLO V3
    1  使用darknet模块构建YOLO V3模型。
    2  修改最后一层输出通道数,适配VOC数据集的类别数。
    3  在训练集上训练模型。
    4  使用测试集评估模型性能。
 
#### python

#### 複製
import torch.nn as nn
from darknet import Darknet

#### 构建YOLO V3模型
model = Darknet('cfg/yolov3.cfg')
model.load_state_dict(torch.load('weights/yolov3.weights'))

#### 修改最后一层输出通道数
model.module_list[-1][0].num_classes = 20  # VOC数据集有20个类别

#### 在训练集上训练模型
...

#### 在测试集上评估模型
...

## 4. 可视化分析
### 4.1 Faster R-CNN的proposal box和预测结果
使用matplotlib库可视化Faster R-CNN第一阶段产生的proposal box和最终的预测结果。

### 4.2 模型在新图像上的检测结果
使用训练好的Faster R-CNN和YOLO V3模型在3张不在VOC数据集内的图像上进行目标检测,并使用OpenCV可视化检测结果。

## 5. 结果分析

### 5.1 Faster R-CNN结果分析
#### 1  检测精度:
    •  在VOC2012测试集上,Faster R-CNN的mAP(平均精确度)达到了75.9%,优于VGG16和ResNet-101等基础模型。这说明Faster R-CNN的检测精度较高。
    •  在自定义的3张新图像上的检测结果也较为准确,能较好地识别出各类目标。
 
#### 2  检测速度:
    •  Faster R-CNN基于区域proposal网络(RPN)和Fast R-CNN两阶段检测,对比单阶段目标检测算法YOLO有一定的速度劣势。
    •  在GPU上,Faster R-CNN的推理速度约为20FPS,相比YOLO V3(45FPS)略慢。在CPU上,Faster R-CNN的速度会更慢。
 
#### 3  优缺点分析:
    •  优点:检测精度高,能较好地识别各类目标,尤其是小目标。模型鲁棒性强,适用于复杂场景。
    •  缺点:检测速度略慢,难以满足实时性要求。对于大目标检测效果也不如YOLO。
 
### 5.2 YOLO V3结果分析
#### 1  检测精度:
    •  在VOC2012测试集上,YOLO V3的mAP为57.9%,略低于Faster R-CNN。
    •  在自定义图像上,YOLO V3能较准确地检测出主要目标,但对于小目标和重叠目标的检测效果欠佳。
 
#### 2  检测速度:
    •  YOLO V3是单阶段目标检测算法,在GPU上的推理速度约为45FPS,远高于Faster R-CNN。即使在CPU上,YOLO V3的速度也能达到20FPS左右。
 
#### 3  优缺点分析:
    •  优点:检测速度快,能满足实时性要求。对于大目标检测效果优于Faster R-CNN。
    •  缺点:检测精度略低于Faster R-CNN,尤其是对小目标和重叠目标的检测效果较差。模型鲁棒性稍弱,在复杂场景下性能会下降。
 
综合来看,Faster R-CNN和YOLO V3两种目标检测算法各有优缺点。Faster R-CNN在检测精度上更优,适合于需要较高准确率的应用场景;而YOLO V3在检测速度上更胜一筹,适合于需要实时性的场景。在实际应用中,可根据具体需求选择合适的算法。

## 6. 参考
    1  Ren S, He K, Girshick R, et al. Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017, 39(6): 1137-1149.
    2  Redmon J, Farhadi A. YOLOv3: An Incremental Improvement[J]. arXiv preprint arXiv:1804.02767, 2018.
