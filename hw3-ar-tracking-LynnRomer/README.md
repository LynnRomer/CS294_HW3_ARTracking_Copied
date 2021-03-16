# HW3: AR Tracking and 3D Reconstruction with OpenCV

For this homework you will be implementing pose estimation of a known image using OpenCV. You will then adapt this pose estimation algorithm to compute a sparse point cloud representation of a scene. This homework is intended to give you a basic introduction to computer vision as it relates to augmented reality. This includes concepts such as camera calibration, feature matching, perspective projection, and 3d point cloud estimation. 

In this homework you will be guided through most of the steps to implement AR pose estimation of a known image, but you will be asked to implement one basic function based on the concepts we introduced in class. You will then use what you have learned to develop a system for sparse point cloud reconstruction of a scene using your mobile device. These steps represent a simplified set of functions that form the basis of modern AR tracking. 


 

## Logistics

After you have accepted the assignment, a seperate repo "hw3-ar-tracking-YourGitID" should have been created. You will push your assignment code to this, and this will be used for grading.

### Deadline

HW2 is due Tuesday 10/06/2020, 11:59PM. Both your code and video need to be turned in for your submission to be complete; HWs which are turned in after 11:59pm will use one of your slip days -- there are no slip minutes or slip hours.

### Academic honesty
Please do not post code to a public GitHub repository, even after the class is finished, since these HWs will be reused both  in the future.

This HW is to be completed individually. You are welcome to discuss the various parts of the HWs with your classmates, but you must implement the HWs yourself -- you should never look at anyone else's code.

## Deliverables:

### 1. Video

You will make a 2 minute video showing off your implementation. You should verbally describe, at a very high level, the concepts used to implement the image pose tracking and 3d reconstruction. You must also include captions corresponding to the audio. This will be an important component of all your homework assignments and your final project so it is best you get this set up early. 

### 2. Code
You will also need to push your project folder to your Github Classroom assignment's repo.


## Before You Start:
For this homework you will need Python and OpenCV. 

### To install python (if you do not have it already installed)

OS X and linux machines, python comes pre-installed. 

In case you do not have it installed, we recommend installing the Anaconda environment.

Download _Anaconda_ with _Python 3.8_ from here - https://www.anaconda.com/products/individual and install it with default options

![inst1.png](/Instructions/inst1.PNG)
![inst2.png](/Instructions/inst2.PNG)
![inst3.png](/Instructions/inst3.PNG)
![inst4.png](/Instructions/inst4.PNG)
![inst5.png](/Instructions/inst5.PNG)

### To install opencv within Anaconda

Choose _Anaconda prompt_ from the start menu, and "Run as Administrator".

![inst6.png](/Instructions/inst6.PNG)

In the Anaconda prompt, we will first create a new Python environment for HW3. This will use Python version 3.7.9
```python
conda create -n hw3env python=3.7
```

Next we will activate this environment. Note you need to run your HW code in this environment, since this is the one in which we wilol be installing openCV
```python
conda activate hw3env
```

Once activated, we will install OpenCV 3.4.7 in it
```python
pip install opencv-contrib-python==3.4.7.28
```

You can choose to run your program in any of the python IDEs in _Anaconda_ such as _Spyder_, but make sure to select the _hw3Env_ before you launch and install the IDE.

![inst7.png](/Instructions/inst7.PNG)

## Install OpenCV without Anaconda
To install OpenCV outside of Anaconda, In command prompt/ terminal run:

```python
pip install opencv-contrib-python==3.4.7.28
```
(you may need to add sudo for unix systems). 

Note: This document was written using OpenCV 3.4.7. Some changes may be required for alternate versions of OpenCV.

## Instructions

Instructions for setting up the starter code are available at the google docs file at https://docs.google.com/document/d/1dNhCcNOpxptHb9oq8h5POKWJUk09LzXlRDgXhROqwts



