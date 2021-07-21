# Device Comparison: Intel NCS2 vs Google Coral Edge TPU
A runthrough of the processes I used to set up Docker using my GPU as well as the Intel NCS2 and Google Coral Edge TPU.

## System
OS: Ubuntu 20.04 LTS  
CPU: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz  
GPU: Nvidia GeForce RTX 2060  
  
## Process
### Docker
1. Uninstall old versions of Docker on the machine  
`sudo apt-get remove docker docker-engine docker.io containerd runc`  
2. Get the newest version of Docker and verify that it is running  
`sudo apt-get update`  
`sudo apt-get install docker-ce docker-ce-cli containerd.io`  
`sudo docker run hello-world`  
3. Create a directory (in this case see `docker_env` in this repository)  
4. Set up the directory in the same structure as this repository  
5. Build the Docker Container  
`docker build -t <name> .`  
6. Start an image of the container  
`docker run -it --gpus all --runtime=nvidia -p 8888:8888 -p 6006:6006 -d -v $(pwd)/notebooks:/notebooks <name>`  
7. Open a browser and access the Jupyter Notebooks through localhost:8888/  
8. Create, train, and save an ML model with Keras `main.ipynb`
### NCS2
1. Install OpenVino Toolkit from https://docs.openvinotoolkit.org/latest/index.html and follow the instructions  
2. Initialize the environment with `source <PATH>/openvino/bin/setupvars.sh` every time a new terminal session is opened  
3. Verify that OpenVino is set up correctly by running one of the demos  
4. Configure USB stick https://software.intel.com/content/www/us/en/develop/articles/get-started-with-neural-compute-stick.html  
5. Configure model optimizer https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_Config_Model_Optimizer.html  
6. Go to the model_optimizer directory and convert the saved Keras model to Intermediate Representation (IR) using the following command  
`mo_tf.py -saved_model <PATH_TO_SAVED_MODEL_DIR> -output_dir <PATH> --input <LAYER_NAME{DTYPE}(flatten_input{f32})> --input_shape <SHAPE([1, 784])>`  
7. Load the converted model into OpenCV in `main.py` in the NCS2 directory  
`python3 main.py`
### Coral TPU
1. Convert and quantize the saved model to TF Lite format and in 8-bit representation using the `model_conversion.ipynb` in the `docker_env` directory  
2. Utilize the TPU using the `libedgetpu` library in the coral directory  
`python3 main.py`

## Run scripts 
Run the `main.py` files in each directory to see the performance of the model using each device
