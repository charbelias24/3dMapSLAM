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


#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
#include "Converter.h"
#include <condition_variable>

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

using namespace ORB_SLAM3;

class PointCloudMapping
{

public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;

    PointCloudMapping( double resolution_ );
    void Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out);
    // Inserting a keyframe updates the map once
    void insertKeyFrame( KeyFrame* kf,cv::Mat& color, cv::Mat& depth, const int seqNum);
    void shutdown();
    void viewer();
    void imageMaskCallback(const sensor_msgs::ImageConstPtr& msgMask);
    void public_cloud( pcl::PointCloud<pcl::PointXYZRGBA> &cloud_kf);
    void generateAndPublishPointCloud(size_t N);

protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, int seqNum);

    PointCloud::Ptr globalMap;

    PointCloud::Ptr KfMap;
    boost::shared_ptr<thread>  viewerThread;
    boost::shared_ptr<thread>  maskSubsThread;
    boost::shared_ptr<thread>  pclThread;
    std::thread  maskThread;

    bool    shutDownFlag    =false;
    std::mutex   shutDownMutex;

    std::condition_variable  keyFrameUpdated;
    std::condition_variable  newMaskArrived;
    std::mutex               keyFrameUpdateMutex;

    // Data to generate point clouds
    std::map<int, cv::Mat>       maskMap;
    std::vector<KeyFrame*>       keyframes;
    std::vector<cv::Mat>         colorImgs;
    std::vector<cv::Mat>         depthImgs;
    std::vector<cv::Mat>         imgMasks;
    std::vector<int>             imgMasksSeq;
    std::mutex                   keyframeMutex;
    std::mutex                   maskMutex;
    uint16_t                lastKeyframeSize =0;

    double resolution = 0.005;
    pcl::VoxelGrid<PointT>  voxel;
    pcl::StatisticalOutlierRemoval<PointT> sor;
};

#endif // POINTCLOUDMAPPING_H
