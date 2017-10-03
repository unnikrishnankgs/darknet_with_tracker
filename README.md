#Real-time traffic pattern collection and analysis model (TPCAM)

The TPCAM traffic surveillance system is implemented as a object tracking by detection model. 
The object detection and classification model used is YOLO, a convolutional neural network. We use darknet neural network framework, to run the model. \
Object tracking functionality is the one major component of a traffic surveillance system. With tracking, the system will learn the motion of vehicles in the video. \
The lane mapping system provides real-time lane information. Thus, the system understand when a vehicle enter or exit lanes, how long a vehicle wait at intersections, and in general the traffic flux.

## Project setup

We used DGX to train our models under evaluation and check their performance.
We had a round of evaluations mainly with pre-trained weights, YOLO and Faster-RCNN.

As we need real-time performance, we are going ahead with YOLO.
We are using the neural network framework: “darknet“ to explore the models, especially YOLO here.

##Darknet (Supporting tracker!!)

Check "BUILD INSTRUCTIONS" section below.

##Why Darknet?
1) The framework is written in C and CUDA.
2) Code is easy to understand.
3) We can easily build the code, tweak the framework for our evaluation needs and application prototyping (especially for AI City Challenge Track 2).

###Darknet Dependencies
1) OpenCV,
2) gstreamer-1.0,
3) gtk-3.0-dev.

Refer to Darknet official site : https://pjreddie.com/darknet/

###Steps to build darknet image
1) Get Darknet source code from git clone : https://github.com/pjreddie/darknet.git
2) In Makefile, change the following macros:
GPU=1
OPENCV=1
CUDANN=1
3) $make

##Object Tracker:

first build the sjtracker: Pre-requisite: OpenCV and OpenCV_contrib. Follow: http://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html to install both of them. [We don't need python for now].

cd objtracker/ ./build.sh

###Installing Dependencies
Check References below (if you’re installing this on personal machines).

In DGX, NVIDIA has provided us an image: “aic/darknet” which might be a fork of https://hub.docker.com/r/jh800222/darknet/.
>NVIDIA DGX Index: ssh aic@10.31.229.235
You can extend the image by:
1) nvidia-docker run -it -v /datasets:/datasets -v /data:/data -v /home/aic:/home/aic aic/darknet
2) exit
3) nvidia-docker ps -a
[Not down the container-ID]
4) nvidia-docker commit ac9b2bceb2b7 yournewcontainername/new_darknet_image_name
Say, example:
nvidia-docker commit ac9b2bceb2b7 test/darknet_test
4) Run your new image:
nvidia-docker run -it -v /datasets:/datasets -v /data:/data -v /home/aic:/home/aic test/darknet_test
5) Remember to commit back your work using steps 2) and 3) above.

##Training YOLO (or any model) using darknet
Even if the document below talks about training YOLO, you could use darknet to train any other model - you’d just want to find or develop .cfg file (similar to yolo-aic.cfg) laying down the neural network in darknet’s “network” format. Already the

###Generating dataset
NVIDIA has provided us the dataset in darknet format at /dataset/aic*-darknet/.

For more info on generating this format,
Dataset folders (generating train.txt and valid.txt)

###Training dataset information (train.txt)
Generate train.txt with details of training data images (path to all jpegs at dataset/train/images/).

To do that: (you can train only 1080p images or choose to train all data)
>$cd /datasets/aic1080-darknet/train/images/

>$find `pwd` > /workspace/darknet/train.txt

>$vi /workspace/darknet/train1080.txt

>And delete the first line “/datasets/aic*-darknet/train/images”

###Validation dataset information (valid.txt)

Generate valid.txt with details of validation data images (path to all jpegs at dataset/val/images/).
To do that: (you can train only 1080p images or choose to train all data)
>$cd /datasets/aic1080-darknet/val/images/

>$find `pwd` > /workspace/darknet/valid.txt

>$vi /workspace/darknet/valid.txt

>And delete the first line “/datasets/aic*-darknet/val/images”

#####Advanced (where we wish to train 540p and 480p images):
Generate valid*.txt in a way similar to the one followed for train*.txt we followed above.
$cat valid540.txt >> valid.txt
$cat valid480.txt >> valid.txt

Now we have train.txt and valid.txt files which are essential input to yolo.c file below which read these and loads images for training.


###Code changes
>Quick Hack: [Use my darknet code at: “https://github.com/unnikrishnankgs/darknet_aic.git”]

*[Still do read the below text to get some detail on training and improving the precision at training or inference; Also go over the Reference links provided].*

 examples/yolo.c has the code to train and validate YOLO’s neural network as defined in a config file, say cfg/yolo-aic.cfg.

Changes required to start training:
1) Change the class labels
2) Paths to train.txt, valid.txt.
3) “jpeg” support:

    In file examples/yolo.c and src/data.c
    * Find the source line: “find_replace(labelpath, ".jpg", ".txt", labelpath);”
    * Just add below this line:
find_replace(labelpath, ".jpeg", ".txt", labelpath);
4) Change in files examples/yolo.c and src/data.c:
“labels” to “annotations” as our labels, the .txt files in darknet format are in a folder named “annotations”.

###Configuring the YOLO model for your own classes

#### **yolo-aic.cfg** file (The Neural Network!)
* Copy cfg/yolo-voc.2.0.cfg to cfg/yolo-aic.cfg and:
* change line batch to batch=64
* change line subdivisions to subdivisions=8
* change line classes=20 to your number of objects
* change line #237 from filters=125 to filters=(classes + 5)*5 (generally this depends on the num and coords, i.e. equal to (classes + coords + 1)*num)

   For example, for 2 objects, your file yolo-aic.cfg should differ from yolo-voc.2.0.cfg in such lines:
   [convolutional] filters=35 \
   [region] classes=2 \
   Also,  there are a number of factors determining each of the hyperparameters that we could play with.
   Read the paper: “https://arxiv.org/pdf/1506.02640.pdf” for a detailed idea.

   *Advanced (when training with multiple resolutions):*
    Change in the config file:
    random=1

#### aic.names
* Create file aic.names in the directory darknet/data, with objects names - each in new line

#### aic.data
* Create file aic.data in the directory darknet/cfg, containing (where classes = number of objects):

  classes= 2 \
  train  = train.txt \
  valid  = valid.txt \
  names = data/aic.names \
  backup = backup/

###Command to Train Yolo

Start training by using pre-trained weights at darknet19_448.conv.23.

> $ ./darknet detector train cfg/aic.data cfg/yolo-aic.cfg /data/team1_darknet/darknet19_448.conv.23

### When to Stop training?
Look out for the error value in the prints during training:
“863: 10.842213, 12.363435 avg, 0.001000 rate, 3.092381 seconds, 55232 images”

Above, 12.363435 is the avg loss (error) - the lower the better.
Stop when this value no longer decrease.

### Inference using darknet
> $ ./darknet detector demo cfg/aic.data cfg/yolo-aic.cfg ~/aic/weights_22Jul/yolo-aic_20000.weights video_file.mp4

If you want to add image-list support, please check: src/demo.c [demo()]

Just use the cvCaptureFromFile() openCV function in a loop. It can open jpeg files (tested).
If you want to use our image list support, please clone the code from "https://github.com/unnikrishnankgs/va/darknet" .

###Improving Object detection
#### Before training:
 set flag random=1 in your .cfg-file

 This will increase precision by training Yolo for different resolutions.
desirable that your training dataset include images with objects at different: scales, rotations, lightings, from different sides.
#### After training - for detection:
Increase network-resolution by set in your .cfg-file (height=608 and width=608) or (height=832 and width=832) or (any value multiple of 32)

This increases the precision and makes it possible to detect small objects:
you do not need to train the network again, just use .weights-file already trained for 416x416 resolution.
if error Out of memory occurs then in .cfg-file you should increase subdivisions=16, 32 or 64.

##References
1) To understand how to provide dataset in darknet format (this part you can skip as AIC provided dataset in darknet format already - check /dataset/aic*-darknet/) and start training.
http://guanghan.info/blog/en/my-works/train-yolo/

2) FAQ, check:
https://groups.google.com/forum/#!forum/darknet

3) For more details on training:
https://github.com/AlexeyAB/darknet
This repo support darknet on windows as well - if you'd like to explore.

4) Further, and precise info on training can be understood here:
https://pjreddie.com/darknet/yolo/

5) Read in full this paper:
https://arxiv.org/pdf/1506.02640.pdf

##Training Log (YOLO with varied resolution images)

Start <11:14PM; 21 July 2017> \
Initial error (loss) is 567.619568 avg \
Learning Rate: 0.0001, Momentum: 0.9, Decay: 0.0005 \
Resizing \
416 \
Loaded: 0.278593 seconds \
Region Avg IOU: 0.039510, Class: 0.070006, Obj: 0.882857, No Obj: 0.605082, Avg Recall: 0.000000,  count: 70 \
Region Avg IOU: 0.087610, Class: 0.048507, Obj: 0.814675, No Obj: 0.612921, Avg Recall: 0.000000,  count: 60\
Region Avg IOU: 0.081044, Class: 0.090126, Obj: 0.841568, No Obj: 0.610928, Avg Recall: 0.028571,  count: 35\
Region Avg IOU: 0.077334, Class: 0.060487, Obj: 0.863747, No Obj: 0.609307, Avg Recall: 0.020833,  count: 48\
Region Avg IOU: 0.091529, Class: 0.105527, Obj: 0.827922, No Obj: 0.608573, Avg Recall: 0.015152,  count: 66\
Region Avg IOU: 0.095050, Class: 0.097945, Obj: 0.785659, No Obj: 0.611849, Avg Recall: 0.000000,  count: 57\
Region Avg IOU: 0.057396, Class: 0.062096, Obj: 0.855653, No Obj: 0.611102, Avg Recall: 0.000000,  count: 46\
Region Avg IOU: 0.062572, Class: 0.061896, Obj: 0.879083, No Obj: 0.606627, Avg Recall: 0.000000,  count: 59\
1: 579.688782, 579.688782 avg, 0.000100 rate, 4.808942 seconds, 64 images\
Loaded: 0.236752 seconds\
Region Avg IOU: 0.039949, Class: 0.058983, Obj: 0.856535, No Obj: 0.565383, Avg Recall: 0.000000,  count: 34 \
Region Avg IOU: 0.057890, Class: 0.089828, Obj: 0.861219, No Obj: 0.555358, Avg Recall: 0.017241,  count: 58\
Region Avg IOU: 0.058388, Class: 0.053862, Obj: 0.856842, No Obj: 0.562076, Avg Recall: 0.000000,  count: 39\
Region Avg IOU: 0.050058, Class: 0.061257, Obj: 0.808184, No Obj: 0.556428, Avg Recall: 0.000000,  count: 52\
Region Avg IOU: 0.113416, Class: 0.093405, Obj: 0.804404, No Obj: 0.565583, Avg Recall: 0.035714,  count: 28\
Region Avg IOU: 0.070091, Class: 0.059165, Obj: 0.777204, No Obj: 0.565168, Avg Recall: 0.017544,  count: 57\
Region Avg IOU: 0.050121, Class: 0.094414, Obj: 0.828460, No Obj: 0.560315, Avg Recall: 0.020000,  count: 50\
Region Avg IOU: 0.063240, Class: 0.057796, Obj: 0.855727, No Obj: 0.559162, Avg Recall: 0.000000,  count: 58\
2: 458.996521, 567.619568 avg, 0.000100 rate, 4.587730 seconds, 128 images\
Loaded: 0.532047 seconds\

###GPU Utilization on DGX:
$ nvidia-smi

Sat Jul 22 06:21:21 2017

````
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 375.66             |   Driver Version: 375.66                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-SXM2...  On   | 0000:00:08.0     Off |                    0 |
| N/A   49C    P0   196W / 300W |   5633MiB / 16276MiB |     99%      Default |
+-------------------------------+----------------------+----------------------+
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID  Type  Process name                               Usage      |
|=============================================================================|
|    0     26340    C   ./darknet                                     5631MiB |
+-----------------------------------------------------------------------------+

===============================================================================
GENERAL INFORMATION
===============================================================================

Our traffic pattern collection is modelled with darknet framework running CNN(YOLO) for detection and our object tracking methodology. \

Support for object tracking (we currently use MEDIAN_FLOW, and optical flow) \
Edge computing support - fasten up real-time data consumption, say for a camera feed using a multi-algorithmic approach \
Read our paper "Real-time traffic pattern collection and analysis model (TPCAM)" for more information \
Further info: unnikrishnankgs@gmail.com \

PLATFORMS: \
Ubuntu; Tested on TX2 and an NVIDIA 1080 GTX enabled PC.

General Pre-req: \

CUDA
CUDNN
OpenCV with opencv_contrib [Mandatory]


===============================================================================
BUILD INSTRUCTIONS
===============================================================================

Object Tracker: 
first build the sjtracker: Pre-requisite: OpenCV and OpenCV_contrib. Follow: http://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html to install both of them. [We don't need python for now]. 

$cd objtracker/ 
$./build.sh 

Darknet (Supporting tracker!!): 

$cd darknet_track/ 
$make

OR (build both together using a single script):

$cd darknet_track/
$./BUILDME.sh

===============================================================================
RUN INSTRUCTIONS
===============================================================================

Download YOLO trained weights: 
https://drive.google.com/open?id=0B-XC86pihjhabFpkdEx5cjhPMUU 
 
Example (Please supply a 480p video - the lane information is scaled for 480p): 
source exports.sh 
./darknet detector demo cfg/aic.data cfg/yolo-aic.cfg your_yolo-aic_final.weights 1080p_WALSH_ST_000.mp4 > out.txt 

If you'd like to supply another video resolution: 
edit src/demo.c line 997, 998 

After the complete video is analysed, traffic patter is dumped in:
data/team1_darknet/$VIDEO_FILE_NAME$/traffic_pattern_timestamp".
- this is a text file in JSON format.

unable to load libraries? 
Dynamic library paths if not in the system defined paths shall be set via LD_LIBRARY_PATH on linux systems (DYLD_LIBRARY_PATH in OS X) See: darknet_track/exports.sh; no need to edit for linux; Just do: $cd darknet_track $source exports.sh

===============================================================================
AUXILIARY INFORMATION
===============================================================================

MORE: Please see: src/demo.c for implementation Use MACRO: DISPLAY_RESULTS if you want to see it in action.
(By default this is enabled) 

Read: include/darknet_exp.h for the interface. 
Call darknet using run_detector_model() function. The data structures should be self-explanatory [We shall add documentation soon..] 


===============================================================================
TROUBLESHOOTING FAQ
===============================================================================
1) Error: "Killed" after the neural network load.
Solution:
Mostly the memory allocation might have overrun the GPU memory capability.
Open: darknet_track/cfg/yolo-aic.cfg
change lines 4,5:
{
height=640
width=640
}
to a lower value; for example:
{
height=416
width=416
}
width and height shall be multiple of 32!

2) OpenCV_contrib build error: dependency opencv_dnn: wouldn't build tracking module.
This is a bug in OpenCV on TX2.
Please clone latest OpenCV code and install following the instructions at http://opencv.org/
> Tutorials > Introduction to OpenCV > Installation in Linux.
