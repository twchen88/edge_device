# ML: docker + NCS2


python3 mo_tf.py --saved_model_dir ~/intel/openvino_2021/NCS2/gpu_test/notebooks/handwritten_model/ --output_dir ~/intel/openvino_2021/NCS2/converted_model --input_shape [1,784]

docker run -it --gpus all --runtime=nvidia -p 8888:8888 -p 6006:6006 -d -v $(pwd)/notebooks:/notebooks test