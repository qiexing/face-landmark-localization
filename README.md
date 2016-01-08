# face-landmark-localization
This is a project predict face landmarks (68 points) and head pose (3d pose, yaw,roll,pitch).


## Install
- [caffe](https://github.com/BVLC/caffe)
- [dlib face detector](http://dlib.net/)<p>
you can down [dlib18.17](http://pan.baidu.com/s/1gey9Wd1) <p>
cd your dlib folder<p>
cd python_example<p>
./compile_dlib_python_module.bat<p>
 add dlib.so to the python path<p>
if using dlib18.18, you can follow the [official instruction](http://dlib.net/)
- opencv<p>

## Usage

- Command : python landmarkPredict.py predictImage  testList.txt<p>
(testList.txt is a file contain the path of the images.)


## Model

- You can download the pre-trained model from [here](http://pan.baidu.com/s/1mhf274c) 

## Result
![](result/1.png)
![](result/2.png)
![](result/3.png)
