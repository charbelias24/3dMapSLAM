#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <string>
#include <unistd.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/core/core.hpp>

// For PCL
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

ros::Publisher pub;

void grabPointcloudWithMask(const sensor_msgs::ImageConstPtr& maskImage, const sensor_msgs::PointCloud2ConstPtr& pointCloud)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrMask;
    try
    {
        cv_ptrMask = cv_bridge::toCvShare(maskImage);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    ROS_INFO("Received new image, pointCloud");
    
    // Create a container for the data.
    sensor_msgs::PointCloud2 output ;

    // Do data processing here...
    output = *pointCloud;

    // Publish the data.
    pub.publish(output);
}


int main(int argc, char **argv)
{
    std::cout << "Starting PointCloud Filtering" << std::endl;
    ros::init(argc, argv, "pointcloud_filtering");
    ros::start();

    ros::NodeHandle nh;
    std::string maskTopicName = "/pointcloud_segmentation/image_mask";
    std::string pointCloudTopicName = "/slam_pointclouds";
    std::string filteredPointCloudTopicName = "/pointcloud_segmentation/filtered_pointcloud";


    std::cout << "Subscribing to" << std::endl;
    std::cout << "\t" << maskTopicName << std::endl;
    std::cout << "\t" << pointCloudTopicName << std::endl;

    message_filters::Subscriber<sensor_msgs::Image> maskSub(nh, maskTopicName, 100);
    message_filters::Subscriber<sensor_msgs::PointCloud2> pointCloudSub(nh, pointCloudTopicName, 100);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), maskSub, pointCloudSub);
    sync.registerCallback(boost::bind(&grabPointcloudWithMask, _1, _2));
    

    // Create a ROS publisher for the output point cloud
    std::cout << "Publishing to\n\t" << filteredPointCloudTopicName << std::endl;
    pub = nh.advertise<sensor_msgs::PointCloud2> (filteredPointCloudTopicName, 1);

    ros::spin();

    ros::shutdown();

    return 0;
}

