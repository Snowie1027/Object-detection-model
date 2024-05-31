# 搜集三张不在VOC数据集内,但包含有VOC中类别物体的图像。
# 加载Faster R-CNN和YOLOv3模型,并进行预处理。

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

# 加载Faster R-CNN模型
faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval()

# 加载YOLOv3模型
yolo = torch.hub.load('ultralytics/yolov3', 'yolov3')
yolo.eval()

# 图像预处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# 分别使用Faster R-CNN和YOLOv3对三张图片进行目标检测,并获得检测结果。

# 加载三张图片
img1 = Image.open('image1.jpg')
img2 = Image.open('image2.jpg')
img3 = Image.open('image3.jpg')

# 使用Faster R-CNN进行检测
with torch.no_grad():
    faster_rcnn_output1 = faster_rcnn([transform(img1)])[0]
    faster_rcnn_output2 = faster_rcnn([transform(img2)])[0]
    faster_rcnn_output3 = faster_rcnn([transform(img3)])[0]

# 使用YOLOv3进行检测
yolo_output1 = yolo(np.array(img1))
yolo_output2 = yolo(np.array(img2))
yolo_output3 = yolo(np.array(img3))
可视化检测结果,并比较两个模型的表现。

# 可视化检测结果
def visualize_detection(img, bboxes, labels, scores):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(img)
    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = [int(x) for x in bbox]
        ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2))
        ax.text(x1, y1, f"{COCO_CLASSES[label]} ({score:.2f})", color='white', fontsize=10, backgroundcolor='red')
    plt.show()

# 在三张图片上比较两个模型的检测结果
visualize_detection(np.array(img1), faster_rcnn_output1['boxes'], faster_rcnn_output1['labels'], faster_rcnn_output1['scores'])
visualize_detection(np.array(img1), yolo_output1.xyxy[0][:, :4], yolo_output1.names, yolo_output1.xyxy[0][:, 4])

visualize_detection(np.array(img2), faster_rcnn_output2['boxes'], faster_rcnn_output2['labels'], faster_rcnn_output2['scores'])
visualize_detection(np.array(img2), yolo_output2.xyxy[0][:, :4], yolo_output2.names, yolo_output2.xyxy[0][:, 4])

visualize_detection(np.array(img3), faster_rcnn_output3['boxes'], faster_rcnn_output3['labels'], faster_rcnn_output3['scores'])
visualize_detection(np.array(img3), yolo_output3.xyxy[0][:, :4], yolo_output3.names, yolo_output3.xyxy[0][:, 4])
