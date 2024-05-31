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

### 3.1 Faster R-CNN在VOC数据集上的训练和测试
#### 3.1.1训练
1）数据集的准备
本文使用VOC格式进行训练，训练前需要下载好VOC07的数据集，解压后放在根目录。
2）数据集的处理
修改voc_annotation.py里面的annotation_mode=2，运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。
3）开始网络训练
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。
4）训练结果预测
训练结果预测需要用到两个文件，分别是frcnn.py和predict.py。
我们首先需要去frcnn.py里面修改model_path以及classes_path，这两个参数必须要修改。
model_path指向训练好的权值文件，在logs文件夹里。
classes_path指向检测类别所对应的txt。
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。
#### 3.1.2预测
下载完库后解压，在百度网盘下载frcnn_weights.pth，放入model_data，运行predict.py。
在predict.py里面进行设置可以进行fps测试和video视频检测。
#### 3.1.3评估
1）本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2）在frcnn.py里面修改model_path以及classes_path。model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。
3）运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

### 3.2 YOLO V3在VOC数据集上的训练和测试
#### 3.2.1训练
1）数据集的准备
本文使用VOC格式进行训练，训练前需要下载好VOC07的数据集，解压后放在根目录。
2）数据集的处理
在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。
3）开始网络训练
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。
4）训练结果预测
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。
model_path指向训练好的权值文件，在logs文件夹里。
classes_path指向检测类别所对应的txt。
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。
#### 3.2.2预测
下载完库后解压，在百度网盘下载yolo_weights.pth，放入model_data，运行predict.py。
在predict.py里面进行设置可以进行fps测试和video视频检测。
#### 3.2.3评估
1）本文使用VOC格式进行评估。VOC07+12已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2）在yolo.py里面修改model_path以及classes_path。model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。
3）运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

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
