import cv2
import numpy as np
import tensorflow as tf

import model_infer


def infer():

    # 加载模型
    interpreter = tf.lite.Interpreter(model_path='yolov5s-fp16-bak.tflite')
    interpreter.allocate_tensors()

    # 获取输入和输出张量的详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 读取本地图片
    image_path = './data/images/hg.jpg'
    image = cv2.imread(image_path)

    # 预处理图像
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    resized_image = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(resized_image.astype(np.float32) / 255.0, axis=0)

    # 设置输入张量的值
    input_tensor_index = input_details[0]['index']
    interpreter.set_tensor(input_tensor_index, input_data)

    # 运行推理
    interpreter.invoke()

    # 获取输出张量的值
    output_tensor_index = output_details[0]['index']
    output_data = interpreter.get_tensor(output_tensor_index)

    # 后处理输出结果
    # 根据输出数据的结构和类型进行相应的后处理操作，例如解析边界框、类别和置信度等信息
    # 解析边界框、类别和置信度
    boxes = output_data[:, :, :4]  # 提取边界框坐标
    classes = output_data[:, :, 4]  # 提取类别
    scores = output_data[:, :, 5]  # 提取置信度

    # 根据置信度阈值进行筛选
    confidence_threshold = 0.5
    filtered_boxes = boxes[scores > confidence_threshold]
    filtered_classes = classes[scores > confidence_threshold]
    filtered_scores = scores[scores > confidence_threshold]

    class_counts = {}
    for cls in filtered_classes:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1

    # 打印不同类别的个数
    for cls, count in class_counts.items():
        print(f"Class: {cls}, Count: {count}")

    # # 打印筛选后的结果
    # for box, cls, score in zip(filtered_boxes, filtered_classes, filtered_scores):
    #     print(f"Class: {cls}, Score: {score}, Box: {box}")


if __name__ == '__main__':
    infer()