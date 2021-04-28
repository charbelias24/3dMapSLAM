/*
 *--------------------------------------------------------------------------------------------------
 * DS-SLAM: A Semantic Visual SLAM towards Dynamic Environments
　*　Author(s):
 * Chao Yu, Zuxin Liu, Xinjun Liu, Fugui Xie, Yi Yang, Qi Wei, Fei Qiao qiaofei@mail.tsinghua.edu.cn
 * Created by Yu Chao@2018.12.03
 * --------------------------------------------------------------------------------------------------
 * DS-SLAM is a optimized SLAM system based on the famous ORB-SLAM2. If you haven't learn ORB_SLAM2 code, 
 * you'd better to be familiar with ORB_SLAM2 project first. Compared to ORB_SLAM2, 
 * we add anther two threads including semantic segmentation thread and densemap creation thread. 
 * You should pay attention to Frame.cc, ORBmatcher.cc, Pointcloudmapping.cc and Segment.cc.
 * 
 *　@article{murORB2,
 *　title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
　*　author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
　* journal={IEEE Transactions on Robotics},
　*　volume={33},
　* number={5},
　* pages={1255--1262},
　* doi = {10.1109/TRO.2017.2705103},
　* year={2017}
 *　}
 * --------------------------------------------------------------------------------------------------
 * Copyright (C) 2018, iVip Lab @ EE, THU (https://ivip-tsinghua.github.io/iViP-Homepage/) and 
 * Advanced Mechanism and Roboticized Equipment Lab. All rights reserved.
 *
 * Licensed under the GPLv3 License;
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * https://github.com/ivipsourcecode/DS-SLAM/blob/master/LICENSE
 *--------------------------------------------------------------------------------------------------
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>

#include "PointCloudMapping.h"

#include <ros/ros.h>
#include "sensor_msgs/PointCloud2.h"
#include <tf/transform_broadcaster.h>
#include <boost/thread/thread.hpp>
#include <boost/chrono.hpp>
#include <tf/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <opencv2/core/core.hpp>

#define MAX_POINTCLOUD_DEPTH 2.0 // in meters

using namespace std;

pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter;
ros::Publisher pclPoint_pub;
ros::Publisher octomap_pub;
ros::Subscriber segmented_sub;
sensor_msgs::PointCloud2 pcl_point;

pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud_kf;

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    this->sor.setMeanK(100);
    this->sor.setStddevMulThresh(0.1);
    viewerThread = boost::make_shared<thread>(bind(&PointCloudMapping::viewer, this));
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, const int seqNum)
{
    cout << "[PCL] Inserting keyframe for seq " << seqNum << endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back(kf);
    colorImgs.push_back(color.clone());
    depthImgs.push_back(depth.clone());
    imgMasksSeq.push_back(seqNum);
    cout << "[PCL] Updated keyframe " << endl;
    keyFrameUpdated.notify_one();
}

pcl::PointCloud<PointCloudMapping::PointT>::Ptr PointCloudMapping::generatePointCloud(KeyFrame *kf, cv::Mat color, cv::Mat depth, int seqNum)
{
    cout << "[PCL] Start generating pointcloud" << seqNum << endl;

    while (maskMap.find(seqNum) == maskMap.end())
    {
        // Mask not found
        // Wait until the mask with the same sequence number
        cout << "[PCL][MASK] Wait, mask " << seqNum << " is not found yet" << endl;
        unique_lock<mutex> lck_maskArrived(maskMutex);
        newMaskArrived.wait(lck_maskArrived);
    }

    cout << "[PCL][MASK] Mask " << seqNum << " is found! " << endl;

    PointCloud::Ptr cloud(new PointCloud());
    cv::Mat *mask = &maskMap[seqNum];

    assert(depth.rows == mask->rows);
    assert(depth.cols == mask->cols);

    // Iterate through all the points in the depth image
    // Only adds points that are within the threshold and are included in the mask
    for (int m = 0; m < depth.rows; m += 1)
    {
        for (int n = 0; n < depth.cols; n += 1)
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > MAX_POINTCLOUD_DEPTH)
                continue;

            // If the flag_exists is true, do not add the points to the point cloud
            bool flag_exist = false;
            int windowSize = 7;
            for (int i = -windowSize; i <= windowSize; i++)
            {
                for (int j = -windowSize; j <= windowSize; j++)
                {
                    int tempx = m + i;
                    int tempy = n + j;

                    if (tempx <= 0)
                        tempx = 0;
                    if (tempx >= (mask->rows - 1))
                        tempx = mask->rows - 1;
                    if (tempy <= 0)
                        tempy = 0;
                    if (tempy >= (mask->cols - 1))
                        tempy = mask->cols - 1;
                    if (!mask->ptr<uint8_t>(tempx)[tempy])
                    {
                        flag_exist = true;
                        break;
                    }
                }
                if (flag_exist)
                    break;
            }

            if (flag_exist)
                continue;

            PointT p;
            p.z = d;
            p.x = (n - kf->cx) * d / kf->fx;
            p.y = (m - kf->cy) * d / kf->fy;

            // Add color to the point based on the rgb image
            p.b = color.ptr<uchar>(m)[n * color.channels()];
            p.g = color.ptr<uchar>(m)[n * color.channels() + 1];
            p.r = color.ptr<uchar>(m)[n * color.channels() + 2];
            if (color.channels() == 4)
                p.a = color.ptr<uchar>(m)[n * color.channels() + 3];

            cloud->points.push_back(p);
        }
    }

    cloud->is_dense = false;
    return cloud;
}

void PointCloudMapping::imageMaskCallback(const sensor_msgs::ImageConstPtr &msgMask)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrMask;
    try
    {
        cv_ptrMask = cv_bridge::toCvShare(msgMask, "mono8");
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    int seqNum = cv_ptrMask->header.seq;

    // cout << "[PCL][MASK] New image received with seq " << seqNum << endl;
    maskMap[seqNum] = cv_ptrMask->image.clone();
    newMaskArrived.notify_one();
}

void PointCloudMapping::generateAndPublishPointCloud(size_t N)
{
    for (size_t i = lastKeyframeSize; i < N; i++)
    {
        PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i], imgMasksSeq[i]);
        PointCloud::Ptr tmp1(new PointCloud());
        voxel.setInputCloud(p);
        voxel.filter(*tmp1);
        p->swap(*tmp1);
        pcl_cloud_kf = *p;

        pcl::toROSMsg(pcl_cloud_kf, pcl_point);
        pcl_point.header.frame_id = "/pointCloudFrame";
        Eigen::Isometry3d T = Converter::toSE3Quat(keyframes[i]->GetPose());
        broadcastTranformMat(T.inverse());

        pclPoint_pub.publish(pcl_point);
        cout << "[PCL] Pointcloud of seq " << imgMasksSeq[i] << " published" << endl;
    }

    lastKeyframeSize = N;
}

void PointCloudMapping::broadcastTranformMat(Eigen::Isometry3d cameraPose)
{
    static tf::TransformBroadcaster transformBroadcaster;

    Eigen::Matrix4d m;

    m << 0, 0, 1, 0,
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 0, 1;

    Eigen::Isometry3d axisTransform(m);
    // Apply axis transformation to the camera pose
    Eigen::Isometry3d finalTransform = axisTransform * cameraPose;

    // Manually create the rotation and translation matrices based on the final transform 
    // in the shape of 4x4 [R T]
    tf::Matrix3x3 rotationMat(
        finalTransform(0, 0), finalTransform(0, 1), finalTransform(0, 2),
        finalTransform(1, 0), finalTransform(1, 1), finalTransform(1, 2),
        finalTransform(2, 0), finalTransform(2, 1), finalTransform(2, 2));

    tf::Vector3 translationMat(
        finalTransform(0, 3), finalTransform(1, 3), finalTransform(2, 3));

    tf::Transform transform;
    transform.setOrigin(translationMat);
    transform.setBasis(rotationMat);

    // Publish the transfrom with the parent frame = /cameraToRobot and create a new child frame /pointCloudFrame
    transformBroadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/cameraToRobot", "/pointCloudFrame"));
}

void PointCloudMapping::viewer()
{
    ros::NodeHandlePtr n = boost::make_shared<ros::NodeHandle>();
    pclPoint_pub = n->advertise<sensor_msgs::PointCloud2>("/slam_pointclouds", 100000);
    segmented_sub = n->subscribe<sensor_msgs::Image>("/pointcloud_segmentation/image_mask", 1000,
                                                     boost::bind(&PointCloudMapping::imageMaskCallback, this, _1));

    while (ros::ok())
    {
        cout << "[PCL] Starting the PCL" << endl;
        {
            unique_lock<mutex> lck_shutdown(shutDownMutex);
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated(keyFrameUpdateMutex);
            keyFrameUpdated.wait(lck_keyframeUpdated);
        }

        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }
        if (N == 0)
        {
            cout << "[PCL] Keyframes miss!" << endl;
            usleep(1000);
            continue;
        }
        
        pclThread = boost::make_shared<thread>(boost::bind(&PointCloudMapping::generateAndPublishPointCloud, this, _1), N);
        pclThread->join();
    }
}
