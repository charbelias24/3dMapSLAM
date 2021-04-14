#!/usr/bin/env python3

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
import rospy
import sys
import roslib
from time import sleep
import pycuda.driver as cuda

from detr.DetrPanopticTRT import DetrPanopticTRT

roslib.load_manifest('pointcloud_segmentation')

class ImageSegmentation:
    def __init__(self):
        image_sub_topic_name = "/zed/zed_node/rgb/image_rect_color"
        image_pub_topic_name = "/pointcloud_segmentation/image_mask"
        engine1_path = "/home/visualbehavior/Documents/surroundnet/weights/detr_panoptic_agv_VGA/detr-panoptic-part1/detr_panoptic_part1_fp16.engine"
        engine2_path = "/home/visualbehavior/Documents/surroundnet/weights/detr_panoptic_agv_VGA/detr-panoptic-part2/detr_panoptic_part2_fp16.engine"
        
        image_size = (376, 672, 3)
        self.model = DetrPanopticTRT(engine1_path, engine2_path, image_size)

        self.bridge = CvBridge()
        self.images_buffer = []

        print ("Subscribing to\n\t", image_sub_topic_name)
        self.image_sub = rospy.Subscriber(
            image_sub_topic_name, Image, self.callback, queue_size=100)

        print ("Publishing to\n\t", image_pub_topic_name)
        self.image_pub = rospy.Publisher(
            image_pub_topic_name, Image, queue_size=100)

        # frame = cv2.imread("/home/visualbehavior/Documents/surroundnet/garage2.png")
        # mask = self.execute_segmentation(frame)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # self.images_buffer.append(frame)
        # self.execute_infer_on_buffer()

    def callback(self, data):
        print ("Received images")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.images_buffer.append(cv_image)
        print("Image Buffer", len(self.images_buffer))
        # (rows, cols, channels) = cv_image.shape
        # frame = cv2.imread("/home/visualbehavior/Documents/surroundnet/garage2.jpg")
        # self.images_buffer.append(frame)

        # masked_image = self.execute_segmentation(frame)

        # masked_image = self.execute_segmentation(cv_image)


    def execute_infer_on_buffer(self):
        if self.images_buffer:
            image = self.images_buffer.pop(0)
            if image is not None:
                mask = self.execute_segmentation(image)
                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(mask, "bgr8"))
                    print ("Publishing mask")
                except CvBridgeError as e:
                    print(e)


    def execute_segmentation(self, image):
        print("Original shape of the iamge is", image.shape)

        image = cv2.resize(image, (672, 376))
        # cv2.imshow("output", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        # print("The shape of the iamge is", image.shape)
        # print("The shape of the iamge is", image.dtype)
        self.model.execute(image)
        boxes, labels, scores, masks = self.model.get_all_output()
 
        # Except the first class and the last class, all other classes are for the ground.
        # You can combine/union all 'ground' mask to have a single mask for the ground 
        # print("-"*10, " class names ", "-"*10)
        # print(self.model.CLASS_NAME)
        # print("-"*10, " bboxes ", "-"*10)
        # [print(i) for i in boxes]
        # print("-"*10, " label indexes ", "-"*10)
        # print(labels)
        # print("-"*10, " scores ", "-"*10)
        # print(scores)
        # print("-"*10, " masks ", "-"*10)
        # print("masks.shape", masks.shape)
        # print("masks.max", masks.max())
        # print("masks.min", masks.min())
        # we should apply sigmoid to mask to have a score ranging from 0 -> 1 for each pixel

        # If you want data stored in GPU, below variables are in type pycuda.driver.DeviceAllocation
        # masks -> model.device_masks
        # boxes -> model.device_out_bbox
        # raw logit scores -> model.device_out_cls

        # For visualization:
        output_image = self.model.get_post_processed_image()

        # cv2.imshow("output", output_image)
        # cv2.waitKey(0)
        return output_image




def main(args):
    image_seg = ImageSegmentation()


    rospy.init_node('image_segmentation', anonymous=True)

    while not rospy.is_shutdown():
        image_seg.execute_infer_on_buffer()

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
