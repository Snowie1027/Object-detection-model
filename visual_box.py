import matplotlib.pyplot as plt
from torchvision.ops import boxes_to_corners

# 假设您已经有了 proposal_boxes 和 predicted_boxes 两个张量
# 分别代表第一阶段的 proposal box 和最终的预测结果

# 将 bounding box 坐标转换为角点坐标
proposal_corners = boxes_to_corners(proposal_boxes)
predicted_corners = boxes_to_corners(predicted_boxes)

# 选择 4 张测试图像进行可视化
for i in range(4):
    # 获取当前图像
    img = your_test_images[i]

    # 绘制图像
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    # 绘制 proposal box
    for corner in proposal_corners[i]:
        x1, y1, x2, y2 = corner
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))

    # 绘制预测结果
    for corner in predicted_corners[i]:
        x1, y1, x2, y2 = corner
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='g', linewidth=2))

    plt.show()
