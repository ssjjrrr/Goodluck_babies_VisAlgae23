from mmdet.apis import init_detector, inference_detector
import os
import json


predictions = []

# 指定模型的配置文件和 checkpoint 文件路径
config_file = 'tianchi/co_dino_5scale_r50_1x_coco.py'
checkpoint_file = 'tianchi/epoch_12.pth'
# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# 测试单张图片并展示结果
dir = '/root/Co-DETR/data/test'
for i in range(701, 1001):
    
    img = f"{i:05}.jpg" # 或者 img = mmcv.imread(img)，这样图片仅会被读一次
    img_path = os.path.join(dir, img)
    result = inference_detector(model, img_path)

    for category_id, category_predictions in enumerate(result):
    # 遍历每个预测结果
        for prediction in category_predictions:
        # 提取边界框和分数
            x_min, y_min, x_max, y_max, score = prediction
        # 计算宽度和高度
            width = x_max - x_min
            height = y_max - y_min
        # 构建字典
            pred_dict = {
                "image_id": i,
                "category_id": category_id,
                "bbox": [float(x_min), float(y_min), float(width), float(height)],
                "score": float(score)
            }
        # 添加到列表
            predictions.append(pred_dict)
    print(f"image{i} ok")
with open('submission.json','w') as f:
    json.dump(predictions, f)


# print(result)
# 在一个新的窗口中将结果可视化P
# model.show_result(img, result)
# 或者将可视化结果保存为图片
# model.show_result(img, result, out_file='tianchi/result706.jpg')
# print(result)
# 测试视频并展示结果
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)