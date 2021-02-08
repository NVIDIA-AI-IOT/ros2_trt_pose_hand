# ros2_trt_pose_hand

In this repository, we build ros2 wrapper for [trt_pose_hand](https://github.com/NVIDIA-AI-IOT/trt_pose_hand) for Real-time hand pose estimation and gesture classification using TensorRT on NVIDIA Jetson Platform.


## Package outputs:
- Hand Pose message with 21 key-points
- Hand Pose detection image message
- `std_msgs` for gesture classification with 6 classes [fist, pan, stop, fine, peace, no hand]
- Visualization markers
- Launch file for RViz

## Requirements:
- ROS 2 Eloquent: <br/>
    - [Install Instructions](https://index.ros.org/doc/ros2/Installation/Eloquent/Linux-Development-Setup/) <br/>
- trt_pose
    - [Dependencies for trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose#step-1---install-dependencies) <br/>
    - [Install trt_pose](https://github.com/NVIDIA-AI-IOT/trt_pose#step-2---install-trt_pose) <br/>
- Gesture Classification
    - scikit-learn ```$ pip3 install -U scikit-learn```

## Build:
- Clone repository under ros2 workspace <br/>
```
$ cd ros2_ws/src/
$ git clone https://github.com/NVIDIA-AI-IOT/ros2_trt_pose_hand.git
```
- Install requirements using ```rosdep``` <br/>
```
$ rosdep install --from-paths src --ignore-src --rosdistro eloquent -y
```
- Build and Install ros2_trt_pose_hand package <br/>
```
$ colcon build
$ source install/local_setup.sh
```

## Run
- Change Power Mode for Jetson
``` sudo nvpmodel -m2 ``` (for Jetson Xavier NX) <br/>
``` sudo nvpmodel -m0 ``` (for Jetson Xavier and Jetson Nano) <br/>
- Keep ```trt_pose_hand``` related model files in ```base_dir```, it should include:<br/>
    - Model files for resnet18 [download link](https://drive.google.com/file/d/1NCVo0FiooWccDzY7hCc5MAKaoUpts3mo/view?usp=sharing)
    - Hand Pose points [json file]()
    - Gesture Classification classes [json file]()
    - Gesture Classification model [included in git]()
- Method 1:<br/>
    - Input Images are captured using ```image_tools``` package <br/>
    ``` ros2 run image_tools cam2image ```
    - Run ```ros2_trt_pose``` node <br/>
        ```
        $ ros2 run ros2_trt_pose_hand  hand-pose-estimation --ros-args -p base_dir:='<absolute-path-to-base_dir>'
        ```
    - Visualize markers <br/>
    ```
    $ ros2 run rviz2 rviz2 launch/hand_pose.rviz
    ```
- Method 2: Use Launch file to each node: <br/>

    - Run using Launch file <br/>
    ```
    $ ros2 launch ros2_trt_pose_hand hand-pose-estimation.launch.py
    ```
    *Note: Update rviz file location in launch file in* ```launch/hand_pose_estimation.launch.py``` <br/>


- For following use separate window for each:<br/>
    - See Pose message <br/>
    ```
    $ source install/local_setup.sh
    $ ros2 run rqt_topic rqt_topic
    ```

## Other related ROS 2 projects
- [ros2_jetson webpage](https://nvidia-ai-iot.github.io/ros2_jetson/)
- [ros2_trt_pose](https://github.com/NVIDIA-AI-IOT/ros2_trt_pose)
- [ros2_torch_trt](https://github.com/NVIDIA-AI-IOT/ros2_torch_trt) : ROS2 Real Time Classification and Detection <br/>
- [ros2_deepstream](https://github.com/NVIDIA-AI-IOT/ros2_deepstream) : ROS2 nodes for DeepStream applications <br/>
- [ros2_jetson_stats](https://github.com/NVIDIA-AI-IOT/ros2_jetson_stats) : ROS 2 package for monitoring and controlling NVIDIA Jetson Platform resources <br/>

