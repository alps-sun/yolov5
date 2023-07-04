import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("yolov5s_saved_model1") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('yolov5s_model.tflite', 'wb') as f:
  f.write(tflite_model)