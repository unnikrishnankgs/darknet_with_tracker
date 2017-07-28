Darknet framework with:
1) Support for object tracking (we currently use MEDIAN_FLOW)
2) Edge computing support - basically to fasten up real-time data consumption, say for a camera feed

for more info:
<unnikrishnankgs@gmail.com>

SUPPORT:
Linux please

BUILD:

General Pre-req:
1) CUDA
2) CUDNN
3) OpenCV [Mandatory]

Object Tracker:
================================================================================================
first build the sjtracker:
Pre-requisite:
OpenCV and OpenCV_contrib.
Follow: http://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html to install both of them.
[We don't need python for now].

cd  objtracker/
./build.sh

Darknet (Supporting tracker!!)
================================================================================================
cd darknet_track/
make

MORE:
Please see:
src/demo.c for implementation
Use MACRO: DISPLAY_RESULTS if you want to see it in action.

Read:
include/darknet_exp.h for the interface.
Call darknet using run_detector_model() function.
The data structures should be self-explanatory 
[We shall add documentation soon..]
