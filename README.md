# ORB-SLAM3 with Segmented 3D Map Construction

- [ORB-SLAM3 with Segmented 3D Map Construction](#orb-slam3-with-segmented-3d-map-construction)
  - [1. Description](#1-description)
  - [2. Prerequisites](#2-prerequisites)
    - [2.1. ORB-SLAM3](#21-orb-slam3)
    - [2.2. ROS](#22-ros)
    - [2.3. CUDA, cuDNN, and TensorRT](#23-cuda-cudnn-and-tensorrt)
    - [2.4. OctoMap](#24-octomap)
    - [2.5. ZED (Optional)](#25-zed-optional)
  - [3. How to build](#3-how-to-build)
  - [4. How to run](#4-how-to-run)
    - [Additional notes](#additional-notes)
  - [5. How it works](#5-how-it-works)
    - [5.1. ZED](#51-zed)
    - [5.2. Segmentation](#52-segmentation)
    - [5.3. SLAM + PointCloud](#53-slam--pointcloud)
    - [5.4. Transformation](#54-transformation)
    - [5.5. OctoMap](#55-octomap)
    - [5.6. RVIZ](#56-rviz)
  - [6. Significant files](#6-significant-files)

*This project was part of a robotics software engineering internship at [Visual Behavior](https://visualbehavior.ai/) - April 2021.*

## 1. Description
The purpose of this software is to use SLAM with RGB-D images to create a 3D colored map of the surroundings, with a possibility of adding and removing static and moving objects, people, floor, etc. by using image segmentation and filtering the point cloud accordingly.

We present an example of constructing a map of the free space on the ground by removing static and moving objects, people, etc. The generated map allows a robot to move freely in this free space. ZED cameras and SVO pre-recorded video were used for demonstration. We used ROS with Python and C++ to combine different modules easily.

The code is modular making it easy to use another image segmentation model to remove or add any object to the 3D map.

It is mainly based on [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3), and we added the map creation using [OctoMap](https://octomap.github.io/) similarly to [DS-SLAM](https://github.com/ivipsourcecode/DS-SLAM). The image segmentation is based on [DETR: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr).

## 2. Prerequisites
We tested the software on **Ubuntu 18.04** with a computer running on **Intel i7**, **16 GB of RAM**, and **RTX 2070**.

### 2.1. ORB-SLAM3
Install all ORB-SLAM3 prerequisites [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)

**IMPORTANT NOTE:** Download **Eigen3.2.x** and **OpenCV3.x** instead of the newer versions. Newer versions may have compatibility issues. You may have to build and compile OpenCV3.x locally: [Installing OpenCV3](https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/)

We used the binary of the ORB Vocabulary instead of the text for faster execution startup. 

**Unzip the `ORBvoc.zip` in the `Vocabulary` directory**, and name it `ORBvoc.bin`

### 2.2. ROS
Tested on ROS-MELODIC: [http://wiki.ros.org/melodic/Installation/Ubuntu](http://wiki.ros.org/melodic/Installation/Ubuntu)

For image segmentation, we need **Python3**. Since ROS’s default python is Python2, we need to add support to Python3.

Follow this article to setup ROS with Python3: [How to setup ROS with Python3](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)

This is an example of how `.bashrc` or `.zshrc` should look like:
```
source /home/${USER}/catkin_ws/devel/setup.zsh
source /home/${USER}/catkin_build_ws/install/setup.zsh
source /home/${USER}/catkin_build_ws/devel/setup.zsh
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/${USER}/ORB_SLAM3/Examples/ROS
```
`catkin_ws` is the recommended ROS directory.
`catkin_build_ws` is the ROS with Python3 support directory.

### 2.3. CUDA, cuDNN, and TensorRT
Tested on CUDA 11.2, cuDNN 8.1.1, TensorRT 7.2.2.

Follow this article to setup the NVIDIA environment: [Setting up your NVIDIA environment](https://blog.jeremarc.com/setup/nvidia/cuda/linux/2020/09/19/nvidia-cuda-setup.html)
### 2.4. OctoMap
Used OctoMap for map generation using ROS and visualized using RVIZ.

2 ROS dependencies are needed: [octomap_mapping](https://github.com/OctoMap/octomap_mapping) and [octomap_rviz_plugins](https://github.com/OctoMap/octomap_rviz_plugins)

We suggest that the installation path of `octomap_mapping` and `octomap_rviz_plugins` be `catkin_ws/src`.

To add color support for the octomap, add `#define COLOR_OCTOMAP_SERVER` into the `OctomapServer.h` at the folder of `octomap_mapping/octomap_server/include/octomap_server`
### 2.5. ZED (Optional)
Tested used ZED cameras and ZED SVO videos. We used a ROS wrapper on top of ZED SDK to easily subscribe to topics create by ZED-Wrapper to get the images and their data.
We used the RGB-D images generated by the ZED stereo cameras.

Note that it is possible to use other types of cameras as long as it publishes the RGB-D image on a ROS topic.
Follow this article to install ZED: [ZED Installation](https://www.stereolabs.com/docs/ros/)
## 3. How to build

Please make sure you have installed all required dependencies.

We provide a script `build.sh` to build **ORB-SLAM3** with **PointCloud**.  Execute:
```
cd ORB_SLAM3
chmod +x build.sh
./build.sh
```
This will create **libORB_SLAM3.so**  at *lib* folder and the executables in *Examples* folder.

Then we need to build the ROS wrapper of ORB-SLAM3 and ImageSegmentation by executing:
```
chmod +x build_ros.sh
./build_ros.sh
```
This targets the directory `Examples/ROS/ORB_SLAM3` and `Examples/ROS/pointcloud_segmentation`

Note: In `build_ros.sh`, you may have to change the path of Python3 paths (`PYTHON_EXECUTABLE`, `PYTHON_INCLUDE_DIR`, `PYTHON_LIBRARY`) mentioned in the article [How to setup ROS with Python3](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)
## 4. How to run
To run the software, you need to run `roscore` (optional) and 2 different bash files: 
1. Run `roscore` in one terminal (optional)
2. In another terminal, execute the `slam_rgbd_zed_ros.sh` file. This executes a ROS launch file `slam_rgbd_zed.launch` with different parameters depending on the ZED camera model.
This ROS launch files runs the ROS Nodes: ORB-SLAM3, ImageSegmentation, OctoMap, Transformations, and RVIZ.
```
chmod +x slam_rgbd_zed_ros.sh
./slam_rgbd_zed_ros.sh zed
```
The first argument is the camera model, with the resolution (default is HD)

Possible arguments are: `zed` `zed-vga` `zed2` `zedm`

1. In another terminal, we need to execute the bash file responsible of publishing images from ZED or SVO files to ROS topics. 
```
chmod +x publish_images_zed.sh
./publish_images_zed.sh zed  # for a connected zed camera with live images
./publish_images_zed.sh zed2 svo_file_path.svo  # for an SVO video recorded with a ZED2 camera
```
Possible arguments:
1. First argument is the camera model: `zed` `zed2` `zedm`
2. Second argument is the SVO file path, if not provided, images from the connected camera will be published


### Additional notes
Order of execution: the order of execution of the bash files doesn't matter, because each one will wait for the other to proceed.

After executing the `slam_rgbd_zed_ros.sh`, 3 new windows should open: image viewer of the current frame, viewer of the trajectory and keyframes, and RVIZ.

Visualizing the OctoMap on RVIZ: in RVIZ, add the `ColorOccupancyGrid` in `octomap_rviz_plugins`, and subscribe to the topic `/octomap_full`. You should see the colored octomap being constructed in real-time. You can display the color of the cell based on the original color, depth, and probability in `Voxel Coloring`.

## 5. How it works
The software is made up of 6 different nodes running in parallel:
### 5.1. ZED
Responsible for taking images from either a camera or an SVO video and publishing the images to a ROS topic.

The ZED camera configurations and parameters (`.yaml` files) are located in `zed/params/`

### 5.2. Segmentation
Responsible for segmenting the RGB images from ZED, and creating a mask based on the segmentations. Then, publishes the mask to a new ROS topic.
Note that the current implementation creates a mask of the ground. This can be easily modified to create a mask of people, etc...
The code is located in `Examples/ROS/pointcloud_segmentation`. 

You can easily use another model by creating a similar node, making sure that you are publishing a mask of the areas to be added to the point cloud.

*For confidential purposes, the class responsible for DETR and segmentation has been removed from the public repo.*

### 5.3. SLAM + PointCloud
1. Runs the official ORB-SLAM3 based on RGB-D input images
2. Additional PointCloudMapper responsible for creating and publishing the point clouds.
  How it works:
   1. Get the RGB-D, KeyFrame, CameraPose, and Sequence number from ORB-SLAM3
   2. Subscribe to the mask ROS topic, and wait for a mask with the same sequence number to be received
   3. Create a point cloud and a transformation matrix (camera to world) based on all the previous data
   4. Publish the point cloud and transformation matrix to 2 new ROS topics
      This code is located in `src/PointCloudMapping.cc`
### 5.4. Transformation
Responsible for handling transformations from the camera to robot, and from the camera to world.

We have 3 frames in the current implementation:
1. `/map` the main frame of reference
2. `/cameraToRobot` the frame which transforms the camera to the robot body (depends on the position and orientation of the camera on the robot)
3. `/pointCloudFrame` the frame of the point cloud, based on the translation and rotation of the camera

Note that frame 3. is a child of frame 2. which is a child of frame 1.
This code is located in `src/PointCloudMapping.cc`
This transformation is applied on the point cloud handled by OctoMap.

**Note:** the `/cameraToRobot` depends on the position and orientation of the camera with respect to the robot. You may need to change this accordingly. This is located in the launch file `/Examples/ROS/ORB_SLAM3/launch/transform.launch`
### 5.5. OctoMap
Responsible for generating an octomap based on the previously published point clouds and transformation matrices.

Note about the **resolution** of PointCloud and OctoMap: you can change the resolution of the published point cloud and octomap by modifying the values in `/Examples/ROS/ORB_SLAM3/launch/octomap.launch` and the value of `PointCloudMapping.Resolution` in the zed configuration `.yaml` files in `/zed/params`.

Make sure that both point cloud and octomap have the same resolution.
### 5.6. RVIZ
Responsible for visualization of the point clouds and the octomap.
## 6. Significant files
- For Main Launch File:
  - `./slam_rgbd_zed.launch`
- For ORB-SLAM3:
  - `./src/System.cc`
  - `./src/Tracking.cc`
  - `./Examples/ROS/ORB_SLAM3/src/ros_rgbd.cc`
- For PointCloud, Segmentation and Transformations:
  - `./src/PointCloudMapping.cc`
  - `./Examples/ROS/pointcloud_segmentation/src/image_segmentation.py`
  - `./Examples/ROS/ORB_SLAM3/launch/transform.launch`
- For Octomap:
  - `./Examples/ROS/ORB_SLAM3/launch/octomap.launch`
  - `/home/${USER}/catkin_ws/src/octomap_mapping/octomap_server/src/OctomapServer.cpp`
- For ZED configurations:
  - `./zed/params`

