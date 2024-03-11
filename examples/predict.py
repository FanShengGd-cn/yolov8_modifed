from PIL import Image
from ultralytics import YOLO

if __name__ == '__main__':
    # 载入一个模型
    # model = YOLO('yolov8n-seg.yaml')  # 从YAML构建一个新模型
    # model = YOLO('yolov8n-seg.pt')    # 载入预训练模型（推荐用于训练）
    # model = YOLO('yolov8n-seg.yaml').load('yolov8n.pt')  # 从YAML构建并传递权重

    # 训练模型
    # results = model.train(data='coco128-seg.yaml', epochs=100, imgsz=640)
    # success = model.export(format="onnx")  # 将模型导出为 ONNX 格式
    # 加载模型
    model = YOLO('D:\Downloads\\ultralytics-main\\ultralytics-main\examples\YOLOv8-Segmentation-ONNXRuntime-Python\\runs')  # 预训练的 YOLOv8n 模型

    # 在图片列表上运行批量推理
    results = model(['D:\Downloadd\\ultralytics-main\\ultralytics-main\examples\YOLOv8-Segmentation-ONNXRuntime-Python\datasets\DataSet2parts\images\\val\\0011.jpg'], stream=True)  # 返回 Results 对象生成器

    # 处理结果生成器
    # 展示结果
    for r in results:
        im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
        im.show()  # 显示图像
        im.save('results.jpg')  # 保存图像