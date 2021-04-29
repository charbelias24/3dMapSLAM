# PointCloud Segmentation

ROS Node responsible for segmenting incoming images and publishing their masks

## Summary:
1. Create a model based on DetrPanopticTRT used for segmenting images 
2. Subscribe to a ROS topic, and save incoming images in a buffer
3. Segment the images in the buffer using the previous model
4. Create a binary mask by combining the generated masks of a single image
5. Publish the binary mask to another ROS topic  


## Building
This is built when running ./build_ros.sh in the project directory. 
Or you can build it manually here by executing
```
mkidr build
cd build
cmake -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so ..
```
Make sure to set the Python3 paths (`PYTHON_EXECUTABLE` `PYTHON_INCLUDE_DIR` `PYTHON_LIBRARY`) following this article: [How to setup ROS with Python3](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674)
