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

PEOPLE_LABEL = 0

def sigmoid(x):
    return 1/(1 + np.exp(-x))


class ImageSegmentation:
    """
    ROS Node responsible for segmenting incoming images and publishing their masks

    Summary:
    1. Create a model based on DetrPanopticTRT used for segmenting images 
    2. Subscribe to a ROS topic, and save incoming images in a buffer
    3. Segment the images in the buffer using the previous model
    4. Create a binary mask by combining the generated masks of a single image
    5. Publish the binary mask to another ROS topic  
    """

    def __init__(self, image_sub_topic_name, mask_pub_topic_name, engine1_path, engine2_path):
        """
        Initialize the ImageSegmentation node and class

        Keyword arguments:
        image_sub_topic_name -- str, name of the ROS topic to subscribe for the incoming images
        mask_pub_topic_name -- str, name of the ROS topic to publish the masks
        engine1_path -- str, path to engine 1 (Backbone + Transformer) for DetrPanopticTRT
        engine2_path -- str, path to engine 2 (Mask Head) for DetrPanopticTRT
        """
        image_size = (376, 672, 3)
        self.model = DetrPanopticTRT(engine1_path, engine2_path, image_size)

        rospy.init_node('image_segmentation', anonymous=True)

        self.bridge = CvBridge()
        self.images_buffer = []

        print("Subscribing to\n\t", image_sub_topic_name)
        self.image_sub = rospy.Subscriber(
            image_sub_topic_name, Image, self._image_received_callback, queue_size=100)

        print("Publishing to\n\t", mask_pub_topic_name)
        self.image_pub = rospy.Publisher(
            mask_pub_topic_name, Image, queue_size=100)

    def _image_received_callback(self, data):
        """
        Convert image message received from the ROS Node to cv2 image, and add to an image buffer

        Keyword arguments:
        data -- ros message containing incoming image
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print("Failed to convert ROS message to cv2 image:", e)

        self.images_buffer.append((cv_image, data.header.seq))

    def _combine_masks(self, masks, labels, image):
        """
        Combine masks of ground areas (excluding people) into one mask and convert it to binary mask

        Keyword arguments:
        masks -- list of masks
        labels -- list of labels 
        image -- input image of the masks

        Return:
        binary mask
        """
        if masks.any():
            img = np.zeros_like(masks[:, :, 0])

            for i in range(0, masks.shape[2]):
                # If the segment at position i has the label people, don't add it to the mask
                if labels[i] == PEOPLE_LABEL:
                    continue

                temp_mask = masks[:, :, i]
                ranged_image = sigmoid(temp_mask) * 255
                img = cv2.add(img, ranged_image)

            _, binary_mask = cv2.threshold(
                img, 255./2., 255, cv2.THRESH_BINARY)

            binary_mask = cv2.resize(
                binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            img = cv2.resize(
                img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            img = np.uint8(binary_mask)
            # cv2.imshow("binary", img)
            # cv2.waitKey(3)
            return img
        return None

    def generate_segmentation_mask(self, image):
        """
        Generate the segmentation of the image using the model,
        and combine the segmented masks to create one binary mask using _combine_masks(masks)

        Keyword arguments:
        image -- input image that needs to be segmented

        Return:
        a binary mask of all segmented masks of input image
        """
        resized_image = cv2.resize(image, (672, 376))
        self.model.execute(resized_image)
        boxes, labels, scores, masks = self.model.get_all_output()
        return self._combine_masks(masks, labels, image)

    def run_once(self):
        """
        Segment the first image in the buffer, create it's binary mask,
        and publish the mask to the relative ROS topic
        """
        if not self.images_buffer:
            return

        image, seq = self.images_buffer.pop(0)
        if image is not None:
            mask = self.generate_segmentation_mask(image)
            if mask is not None and mask.any():
                try:
                    ros_mask = self.bridge.cv2_to_imgmsg(mask, "mono8")
                    # The mask should have the same seq number as the image
                    ros_mask.header.seq = seq
                    self.image_pub.publish(ros_mask)
                    print("Publishing mask", ros_mask.header.seq)

                except CvBridgeError as e:
                    print(e)


def main(args):
    image_sub_topic_name = "/zed/zed_node/rgb/image_rect_color"
    mask_pub_topic_name = "/pointcloud_segmentation/image_mask"
    engine1_path = "/home/visualbehavior/Documents/surroundnet/weights/detr_panoptic_agv_VGA/detr-panoptic-part1/detr_panoptic_part1_fp16.engine"
    engine2_path = "/home/visualbehavior/Documents/surroundnet/weights/detr_panoptic_agv_VGA/detr-panoptic-part2/detr_panoptic_part2_fp16.engine"

    image_seg = ImageSegmentation(image_sub_topic_name, mask_pub_topic_name,
                                  engine1_path, engine2_path)

    rate = rospy.Rate(60) # 50Hz

    while not rospy.is_shutdown():
        image_seg.run_once()
        # rate.sleep()

    try:
        rospy.spin()

    except KeyboardInterrupt:
        print("Shutting down")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv)
