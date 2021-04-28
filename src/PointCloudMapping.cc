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

using namespace std;

bool transformLocally;
bool testingTransform;

pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter;
ros::Publisher pclPoint_pub;
ros::Publisher octomap_pub;
ros::Subscriber segmented_sub;
sensor_msgs::PointCloud2 pcl_point;

pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud_kf;

PointCloudMapping::PointCloudMapping(double resolution_)
{
    transformLocally = false;
    testingTransform = false;
    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    this->sor.setMeanK(100);
    this->sor.setStddevMulThresh(0.1);
    globalMap = boost::make_shared<PointCloud>();
    KfMap = boost::make_shared<PointCloud>();
    viewerThread = boost::make_shared<thread>(bind(&PointCloudMapping::viewer, this));
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    maskSubsThread->join();
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame *kf, cv::Mat &color, cv::Mat &depth, const int seqNum)
{
    cout << "[MASK] Inserting keyframe for seq " << seqNum << endl;
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
    cout << "[MASK] Start generating pointcloud" << seqNum << endl;

    while (maskMap.find(seqNum) == maskMap.end())
    {
        // Mask not found 0
        // Wait until the mask with the same timestamp
        cout << "[MASK] Wait, mask " << seqNum << " is not found yet" << endl;
        unique_lock<mutex> lck_maskArrived(maskMutex);
        newMaskArrived.wait(lck_maskArrived);
    }

    PointCloud::Ptr tmp(new PointCloud());

    cout << "[MASK] Mask " << seqNum << " is found! " << endl;
    cv::Mat *mask = &maskMap[seqNum];

    double minT, maxT;
    cv::minMaxIdx(*mask, &minT, &maxT);

    // Point cloud is null ptr
    assert(depth.rows == mask->rows);
    assert(depth.cols == mask->cols);

    for (int m = 0; m < depth.rows; m += 1)
    {
        for (int n = 0; n < depth.cols; n += 1)
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 2.0)
                continue;

            int flag_exist = 0;

            for (int i = -7; i <= 7; i++)
            {
                for (int j = -7; j <= 7; j++)
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
                        flag_exist = 1;
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

            // Deal with color
            p.b = color.ptr<uchar>(m)[n * color.channels()];
            p.g = color.ptr<uchar>(m)[n * color.channels() + 1];
            p.r = color.ptr<uchar>(m)[n * color.channels() + 2];
            if (color.channels() == 4)
                p.a = color.ptr<uchar>(m)[n * color.channels() + 3];

            tmp->points.push_back(p);
        }
    }

    // cout << "[PCL] Generate point cloud for kf " << kf->mnId << ", size=" << cloud->points.size() << endl;

    if (transformLocally)
    {
        Eigen::Isometry3d T = Converter::toSE3Quat(kf->GetPose());
        PointCloud::Ptr cloud(new PointCloud);
        pcl::transformPointCloud(*tmp, *cloud, T.inverse().matrix());
        cloud->is_dense = false;
        return cloud;
    }
    else
    {
        tmp->is_dense = false;
        return tmp;
    }
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

    cout << "[MASK] New image received with seq " << seqNum << endl;
    maskMap[seqNum] = cv_ptrMask->image.clone();
    newMaskArrived.notify_one();
}

void PointCloudMapping::generateAndPublishPointCloud(size_t N)
{
    cout << "[MASK] When generating, num of stored seq " << imgMasksSeq.size() << endl;
    for (size_t i = lastKeyframeSize; i < N; i++)
    {
        PointCloud::Ptr p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i], imgMasksSeq[i]);

        if (transformLocally)
        {
            *KfMap += *p;
            *globalMap += *p;

            PointCloud::Ptr tmp1(new PointCloud());
            voxel.setInputCloud(KfMap);
            voxel.filter(*tmp1);
            KfMap->swap(*tmp1);
            pcl_cloud_kf = *KfMap;

            Cloud_transform(pcl_cloud_kf, pcl_filter);
            pcl::toROSMsg(pcl_filter, pcl_point);

            pcl_point.header.frame_id = "/pointCloud";
        }
        else
        {
            PointCloud::Ptr tmp1(new PointCloud());
            voxel.setInputCloud(p);
            voxel.filter(*tmp1);
            p->swap(*tmp1);
            pcl_cloud_kf = *p;

            if (testingTransform)
            {
                Eigen::Isometry3d T = Converter::toSE3Quat(keyframes[i]->GetPose());
                testing_transform(pcl_cloud_kf, pcl_filter, T.inverse());
                pcl::toROSMsg(pcl_filter, pcl_point);
                pcl_point.header.frame_id = "/pointCloud";
            }
            else
            {
                pcl::toROSMsg(pcl_cloud_kf, pcl_point);
                pcl_point.header.frame_id = "/pointCloudFrame";
                Eigen::Isometry3d T = Converter::toSE3Quat(keyframes[i]->GetPose());
                broadcastTranformMat(T.inverse());
            }
        }

        pclPoint_pub.publish(pcl_point);
        cout << "[PCL] Keyframe map published" << endl;
    }

    lastKeyframeSize = N;
}

void PointCloudMapping::broadcastTranformMat(Eigen::Isometry3d transformation)
{
    static tf::TransformBroadcaster transformBroadcaster;

    Eigen::Matrix4d m;

    m << 0, 0, 1, 0,
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 0, 1;

    Eigen::Isometry3d axisTransform(m);
    Eigen::Isometry3d finalTransform = axisTransform * transformation;

    tf::Matrix3x3 rotationMat(
        finalTransform(0, 0), finalTransform(0, 1), finalTransform(0, 2),
        finalTransform(1, 0), finalTransform(1, 1), finalTransform(1, 2),
        finalTransform(2, 0), finalTransform(2, 1), finalTransform(2, 2));

    tf::Vector3 translationMat(
        finalTransform(0, 3), finalTransform(1, 3), finalTransform(2, 3));

    tf::Transform transform;
    transform.setOrigin(translationMat);
    transform.setBasis(rotationMat);
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
            cout << "[MASK] keyFrameUpdated resumed" << endl;
        }

        size_t N = 0;
        {
            unique_lock<mutex> lck(keyframeMutex);
            N = keyframes.size();
        }
        if (N == 0)
        {
            cout << "[MASK] Keyframes miss!" << endl;
            usleep(1000);
            continue;
        }
        KfMap->clear();

        cout << "[MASK] Will create new thread for generation" << endl;
        pclThread = boost::make_shared<thread>(boost::bind(&PointCloudMapping::generateAndPublishPointCloud, this, _1), N);
        pclThread->join();
        cout << "[MASK] New thread of generateAndPublishCloud has been joined" << endl;
    }
}

void PointCloudMapping::public_cloud(pcl::PointCloud<pcl::PointXYZRGBA> &cloud_kf)
{
    cloud_kf = pcl_cloud_kf;
}

void PointCloudMapping::Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA> &source, pcl::PointCloud<pcl::PointXYZRGBA> &out)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered;
    Eigen::Matrix4f m;

    m << 0, 0, 1, 0,
        -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 0, 1;
    Eigen::Affine3f transform(m);
    cout << "Affine3f " << endl
         << transform.matrix() << endl;

    Eigen::Isometry3f axisTransform(m);
    cout << "Isometry3d " << endl
         << axisTransform.matrix() << endl;

    pcl::transformPointCloud(source, out, transform);
}
