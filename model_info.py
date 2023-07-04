import tensorflow.lite as tflite

def model_information():

    # 加载模型
    interpreter = tflite.Interpreter(model_path="yolov5s-fp16-bak.tflite")
    interpreter.allocate_tensors()

    # 获取模型信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # input info
    print("input layer dateails:")
    for input_layer in input_details:
        print(f"Name: {input_layer['name']}")
        print(f"Type : {input_layer['dtype']}")
        print(f"Shape: {input_layer['shape']}")
    # 输出层信息
    print("Output layer details:")
    for output_layer in output_details:
        print(f"Name: {output_layer['name']}")
        print(f"Type: {output_layer['dtype']}")
        print(f"Shape: {output_layer['shape']}")
        output_tensor = interpreter.get_tensor(output_details[0]['index'])


if __name__ == '__main__':
    model_information()
