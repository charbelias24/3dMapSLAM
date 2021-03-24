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


#include "PointCloudMapping.h"

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include <tf/transform_broadcaster.h> 

pcl::PointCloud<pcl::PointXYZRGBA> pcl_filter; 
ros::Publisher pclPoint_pub;
ros::Publisher octomap_pub;
sensor_msgs::PointCloud2 pcl_point;

pcl::PointCloud<pcl::PointXYZRGBA> pcl_cloud_kf;

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    this->sor.setMeanK(50);                                
    this->sor.setStddevMulThresh(1.0);                    
    globalMap = boost::make_shared< PointCloud >( );
    KfMap = boost::make_shared< PointCloud >( );
    viewerThread = boost::make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
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

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{
    
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );
    cout << "[PCL] Updated keyframe " << endl; 
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth)
{

    PointCloud::Ptr tmp( new PointCloud() );
    // Point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=1)
    {
        for ( int n=0; n<depth.cols; n+=1)
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 4.0)
                continue;

            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;

            // Deal with color
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;

    cout<<"[PCL] Generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


void PointCloudMapping::viewer()
{

    ros::NodeHandle n;
    pclPoint_pub = n.advertise<sensor_msgs::PointCloud2>("/slam_pointclouds", 100000);
    ros::Rate r(5);
    while(1)
    {
        cout << "[PCL] HWEW" << endl;
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }

        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        if(N==0)
	    {
	        cout<<"[PCL] Keyframes miss!"<<endl;
            usleep(1000);
	        continue;
	    }
        KfMap->clear();
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            PointCloud::Ptr p = generatePointCloud( keyframes[i],colorImgs[i], depthImgs[i] );
	        *KfMap += *p;
	        *globalMap += *p;	    
        }
	
	    PointCloud::Ptr tmp1(new PointCloud());
        voxel.setInputCloud( KfMap );
        voxel.filter( *tmp1 );
        KfMap->swap( *tmp1 );	
        pcl_cloud_kf = *KfMap;	

	    Cloud_transform(pcl_cloud_kf,pcl_filter);
	    pcl::toROSMsg(pcl_filter, pcl_point);
	    pcl_point.header.frame_id = "/pointCloud";
	    pclPoint_pub.publish(pcl_point);
        lastKeyframeSize = N;
	    cout << "[PCL] Keyframe map publish time ="<<endl;
    }

}

void PointCloudMapping::public_cloud( pcl::PointCloud< pcl::PointXYZRGBA >& cloud_kf )
{
	cloud_kf =pcl_cloud_kf; 
}

void PointCloudMapping::Cloud_transform(pcl::PointCloud<pcl::PointXYZRGBA>& source, pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_filtered;
	Eigen::Matrix4f m;

	m<< 0,0,1,0,
	    -1,0,0,0,
		0,-1,0,0;
	Eigen::Affine3f transform(m);
	pcl::transformPointCloud (source, out, transform);
}
