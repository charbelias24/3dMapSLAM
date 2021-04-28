#!/usr/bin/env python3

from pycuda import gpuarray
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import numpy as np
import cv2

np.random.seed(0)

PERSON_THRESHOLD = 0.5
FLOOR_THRESHOLD = 0.55
BLOCKSIZE = 256
CLASS_COLOR_MAP = np.random.randint(0, 255, (300, 3))
CLASS_COLOR_MAP[0] = [238,130,238] # person color
# for i in range(300):
#     CLASS_COLOR_MAP[i] = [238,130,238]

# CLASS_COLOR_MAP[1] = [152,251,152] # floor color
GPU_CLASS_COLOR_MAP = gpuarray.to_gpu(CLASS_COLOR_MAP.astype(np.uint8)) # uint16 == unsigned short

gpu_channel_avg = gpuarray.to_gpu(np.array([0.485, 0.456, 0.406], dtype = np.float32))
gpu_channel_std = gpuarray.to_gpu(np.array([0.229, 0.224, 0.225], dtype = np.float32))
# print("GPU_CLASS_COLOR_MAP.dtype", GPU_CLASS_COLOR_MAP.dtype)


cuFunctions = SourceModule(
    """
    #include <stdio.h>
    #define PERSON_ID 0
    #define PERSON_THRESHOLD 0.5
    #define FLOOR_THRESHOLD 0.76

    __global__ void normalize_image_OLD(float*  img, const float* channel_avg, const float* channel_std)
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        img[i] = (img[i]/255 - channel_avg[i%3])/channel_std[i%3];
    }

    __global__ void sigmoid(
        float* arr, const int nb_element
    )
    {
        // use 1D block in 1D grid
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if( i > nb_element) return;
        arr[i] = 1/(1 + exp(-1.0*arr[i]));
    }

    __global__ void normalize_image(
        float*  img, const float* channel_avg, const float* channel_std, 
        const int H, const int W, const int C
    )
    {
        // use 3D blocks in a 2D grid 
        // img (H, W, C)
        // channel_*** (C, 1)

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;
        if(x >= W) return;
        if(y >= H) return;
        if(z >= C) return;
        const int i = z + x*C + y*W*C;
        img[i] = (img[i]/255 - channel_avg[z])/channel_std[z];
    }

    __global__ void nearest_resize_batch_masks(
        float* dst_img, const float* src_img,
        const int dst_w, const int dst_h, 
        const int src_w, const int src_h, const int N
    )
    {
        // src_img (N, h, w)
        const int y = threadIdx.x + blockIdx.x * blockDim.x;
        const int k = threadIdx.y + blockIdx.y * blockDim.y;
        const int x = threadIdx.z + blockIdx.z * blockDim.z;
        if(x >= dst_w) return;
        if(y >= dst_h) return;
        if(k >= N) return;
        const int i = k*dst_w*dst_h + dst_w*y + x;

        float x_ratio = src_w/(float)dst_w;
        float y_ratio = src_h/(float)dst_h;
        
        int src_x, src_y, src_i;
        src_x = x * x_ratio;
        src_y = y * y_ratio;
        src_i = k*src_w*src_h + src_w*src_y + src_x;
        dst_img[i] = src_img[src_i];

        // printf("i=%d n=%d dst_x=%d dst_y=%d src_x=%d src_y=%d val=%f\\n", i, n, dst_x, dst_y, src_x, src_y, dst_img[i]);
        // printf("i=%d n=%d val=%f\\n", i, n, dst_img[i]);
    }

    __global__ void bilinear_resize_masks(
        float* dst_img, const float* src_img,
        const int dst_w, const int dst_h, 
        const int src_w, const int src_h, const int N
    )
    {
        // src_img (N, h, w)
        const int y = threadIdx.x + blockIdx.x * blockDim.x;
        const int k = threadIdx.y + blockIdx.y * blockDim.y;
        const int x = threadIdx.z + blockIdx.z * blockDim.z;

        if(x >= dst_w) return;
        if(y >= dst_h) return;
        if(k >= N) return;

        const int i = k*dst_w*dst_h + dst_w*y + x;

        const float x_ratio = src_w/(float)dst_w;
        const float y_ratio = src_h/(float)dst_h;

        const int src_x_low = floor(x * x_ratio);
        const int src_y_low = floor(y * y_ratio);
        const int src_x_high = src_x_low + 1;
        const int src_y_high = src_y_low + 1;

        const float x_weight = x_ratio * x - src_x_low;
        const float y_weight = y_ratio * y - src_y_low;

        float a = src_img[k*src_w*src_h + src_w*src_y_low + src_x_low];
        float b = src_img[k*src_w*src_h + src_w*src_y_low + src_x_high];
        float c = src_img[k*src_w*src_h + src_w*src_y_high + src_x_low];
        float d = src_img[k*src_w*src_h + src_w*src_y_high + src_x_high];

        dst_img[i] = a * (1 - x_weight) * (1 - y_weight) 
                    + b * x_weight * (1 - y_weight) 
                    + c * y_weight * (1 - x_weight) 
                    + d * x_weight * y_weight;
    }

    __global__ void create_panoptic_coco(
            float* img, const unsigned short* class_ids,
            const unsigned char* colors, const float* resized_masks, const unsigned short* bbox_xy,
            const int H, const int W, const int C, const int nbox, const float threshold
    )
    {
        // img (H, W, C)
        // class_ids (nbox, 1)
        // colors (256, 1)
        // resized_masks (nbox, H, W)
        // bbox_xy (nbox, 4)

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;
        if(x >= W) return;
        if(y >= H) return;
        if(z >= C) return;
        const int i = z + x*C + y*W*C;
        int i_mask, cls;
        
        for(int k=0; k<nbox; k++)
        {
            i_mask = k*H*W + y*W + x;
            cls = class_ids[k];
            if ( resized_masks[i_mask] > threshold )
            {
                img[i] = 0.4*img[i] + 0.6*colors[cls*3 + z];
                break;
            }
            
        }
    }

    __global__ void create_panoptic(
            float* img, const unsigned short* class_ids,
            const unsigned char* colors, const float* resized_masks, const unsigned short* bbox_xy,
            const int H, const int W, const int C, const int nbox, const float person_threshold, const float floor_threshold
    )
    {
        // img (H, W, C)
        // class_ids (nbox, 1)
        // colors (256, 1)
        // resized_masks (nbox, H, W)
        // bbox_xy (nbox, 4)

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;
        if(x >= W) return;
        if(y >= H) return;
        if(z >= C) return;
        const int i = z + x*C + y*W*C;
        int i_mask, cls;
        float threshold;
        
        for(int k=0; k<nbox; k++)
        {
            i_mask = k*H*W + y*W + x;
            cls = class_ids[k];
            if(cls != PERSON_ID)
            {
                cls = 1; // we treat all types of floor as an unique class, here we choose class 1
                threshold = floor_threshold;
                if ( resized_masks[i_mask] > threshold )
                {
                    img[i] = 0.4*img[i] + 0.6*colors[cls*3 + z];
                    break;
                }
            }
            else
            {
                threshold = person_threshold;
                if ( resized_masks[i_mask] > threshold )
                {
                    img[i] = 0.4*img[i] + 0.6*colors[cls*3 + z];
                    break;
                }
            } 
        }
    }

    __device__ float dist_to_proba(const float dist, const int dist_max)
    {
        if(dist > dist_max) return 0.0;
        return (dist_max - dist)/(float)dist_max;
    }

    __global__ void create_panoptic_combined(
            float* img, const unsigned short* class_ids,
            const unsigned char* colors, const float* resized_masks, const unsigned short* bbox_xy,
            const int H, const int W, const int C, const int nbox, const float person_threshold, const float floor_threshold,
            const float* dist_plan
    )
    {
        // img (H, W, C)
        // class_ids (nbox, 1)
        // colors (256, 1)
        // resized_masks (nbox, H, W)
        // bbox_xy (nbox, 4)

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;
        if(x >= W) return;
        if(y >= H) return;
        if(z >= C) return;
        const int i = z + x*C + y*W*C;
        int i_mask, cls, i_dist;
        float threshold, floor_proba;
        
        
        for(int k=0; k<nbox; k++)
        {
            i_mask = k*H*W + y*W + x;
            cls = class_ids[k];
            i_dist = x + y*W;
            if(cls != PERSON_ID)
            {
                cls = 1; // we treat all types of floor as an unique class, here we choose class 1
                threshold = floor_threshold;
                floor_proba = (resized_masks[i_mask] + dist_to_proba(dist_plan[i_dist], 0.05));
                if ( floor_proba > threshold )
                {
                    img[i] = 0.4*img[i] + 0.6*colors[cls*3 + z];
                    break;
                }
            }
            else
            {
                threshold = person_threshold;
                if ( resized_masks[i_mask] > threshold )
                {
                    img[i] = 0.4*img[i] + 0.6*colors[cls*3 + z];
                    break;
                }
            } 
        }
    }


    __global__ void denormalize_image(
        float* img, const float* channel_avg, const float* channel_std,
        const int H, const int W, const int C
    )
    {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;
        if(x >= W) return;
        if(y >= H) return;
        if(z >= C) return;
        const int i = z + x*C + y*W*C;
        img[i] = (img[i]*channel_std[z] + channel_avg[z])*255.0;
    }

    __global__ void cvtBGR2RGB(
        float* img, const int H, const int W, const int C
    )
    {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;
        if(z >= 1) return; // each kernal calculate all 3 channels 
        if(x >= W) return;
        if(y >= H) return;
        const int i = z + x*C + y*W*C;
        float blue, red;
        blue = img[i];
        red = img[i + 2];
        img[i] = red;
        img[i + 2] = blue;
    }

    __global__ void astype_uint8(
        unsigned char* out, const float* in
    )
    {   
        // just use 1D block in 1D grid
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        out[i] = (unsigned char)(in[i]);
    }

    __global__ void uint8_to_float(
        float* out, const unsigned char* in
    )
    {   
        // just use 1D block in 1D grid
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        out[i] = (float)(in[i]);
    }
    
    __global__ void bilinear_resize_image(
        unsigned char* dst_img, const unsigned char* src_img,
        const int dst_w, const int dst_h, 
        const int src_w, const int src_h, const int C
    )
    {
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        const int z = threadIdx.z + blockIdx.z * blockDim.z;

        if(x >= dst_w) return;
        if(y >= dst_h) return;
        if(z >= C) return;

        const int i = z + x*C + y*dst_w*C;

        const float x_ratio = src_w/(float)dst_w;
        const float y_ratio = src_h/(float)dst_h;

        const int src_x_low = floor(x * x_ratio);
        const int src_y_low = floor(y * y_ratio);
        const int src_x_high = src_x_low + 1;
        const int src_y_high = src_y_low + 1;

        const float x_weight = x_ratio * x - src_x_low;
        const float y_weight = y_ratio * y - src_y_low;

        unsigned char a = src_img[z + src_x_low*C + src_y_low*C*src_w];
        unsigned char b = src_img[z + src_x_high*C + src_y_low*C*src_w];
        unsigned char c = src_img[z + src_x_low*C + src_y_high*C*src_w];
        unsigned char d = src_img[z + src_x_high*C + src_y_high*C*src_w];

        dst_img[i] = a;
        // dst_img[i] = a * (1 - x_weight) * (1 - y_weight) + b * x_weight * (1 - y_weight) + c * y_weight * (1 - x_weight) + d * x_weight * y_weight;
    }

    __global__ void matmul(
        float* R, const float* A, const float* B,
        const int nb_row_A, const int nb_col_A, 
        const int nb_row_B, const int nb_col_B
    )
    {
        // R = A*B
        // use 2D blocks
        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;

        if(x >= nb_col_B) return;
        if(y >= nb_row_A) return;

        const int i = x + y*nb_col_B;
        float sum = 0.0;

        for(int k=0; k < nb_col_A; k++)
        {
            sum += A[y*nb_col_A + k] * B[k*nb_col_A + x];
        }
        R[i] = sum;
    }

    __global__ void get_floor_pointcloud_filter(
        unsigned char* pc_filter, const float* resized_masks, const unsigned short* labels,
        const int nbox, const int img_h, const int img_w
    )
    {
        // pc_filter (H*W, 1)
        // resized_masks (nbox, H, W)
        // labels (nbox, 1)
        // Use 2D blocks

        const int x = threadIdx.x + blockIdx.x * blockDim.x;
        const int y = threadIdx.y + blockIdx.y * blockDim.y;
        if(x >= img_w) return;
        if(y >= img_h) return;
        const int i = x + y*img_w;
        int cls, i_mask;
        pc_filter[i] = 0;

        for(int k=0; k<nbox; k++)
        {
            cls = labels[k];
            i_mask = k*img_h*img_w + y*img_w + x;
            if(cls != PERSON_ID)
            {
                if( resized_masks[i_mask] > FLOOR_THRESHOLD)
                {
                    pc_filter[i] = 2;
                    break;
                }
            }
        }
    }

    __global__ void pointcloud_to_birdview(
        unsigned char* birdview, 
        const float* pointcloud, const unsigned char* pointcloud_rgb, const unsigned char* pointcloud_filter,
        const int birdview_h, const int birdview_w, const int img_h, const int img_w,
        const float range_x_min, const float range_x_max,
        const float range_z_min, const float range_z_max
    )
    {
        // birdview (h, w, 3)
        // pointcloud (H*W, 3)
        // pointcloud_rgb (H*W, 3)
        // pointcloud_filter (H*W, 1)
        // use 1D blocks of size H*W, 1, 1

        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i >= img_h*img_w) return;
        const float x_relative = (pointcloud[i*3] - range_x_min)/(range_x_max - range_x_min);
        const float y_relative = (pointcloud[i*3 + 2] - range_z_min)/(range_z_max - range_z_min);
        const int x = x_relative*(float)birdview_w;
        const int y = birdview_h - y_relative*birdview_h;
        
        // clip value
        if( (x > 0) && (x < birdview_w - 1) )
        {
            if( (y > 0) && (y < birdview_h - 1) )
            {

                // TODO : improve the filter
                if(pointcloud_filter[i] != 0)
                {
                    for(int k=0; k<3; k++)
                    {
                        birdview[k + x*3 + y*birdview_w*3] = pointcloud_rgb[i*3 + k];
                    }
                }
                else
                {
                    for(int k=0; k<3; k++)
                    {
                        birdview[k + x*3 + y*birdview_w*3] = 0;
                    }
                }
            }
        }
    }

    __global__ void clean_birdview(
        unsigned char* birdview, const int birdview_h, const int birdview_w
    )
    {
        const int i = threadIdx.x + blockIdx.x * blockDim.x;
        if(i >= birdview_h*birdview_w*3) return;
        birdview[i] = 0;
    }
    """
)

cuNormalize_image = cuFunctions.get_function("normalize_image")
cuResize_batch_masks = cuFunctions.get_function("nearest_resize_batch_masks")
cuCreate_panoptic = cuFunctions.get_function("create_panoptic")
cuCreate_panoptic_coco = cuFunctions.get_function("create_panoptic_coco")
cuCreate_panoptic_combined = cuFunctions.get_function("create_panoptic_combined")
cuDenormalize_image = cuFunctions.get_function("denormalize_image")
cuCvtBGR2RGB = cuFunctions.get_function("cvtBGR2RGB")
cuAstype_uint8 = cuFunctions.get_function("astype_uint8")
cuUint8_to_float32 = cuFunctions.get_function("uint8_to_float")
cuSigmoid = cuFunctions.get_function("sigmoid")
cuBilinear_resize_masks = cuFunctions.get_function("bilinear_resize_masks")
cuBilinear_resize_image = cuFunctions.get_function("bilinear_resize_image")
cuMatmul = cuFunctions.get_function("matmul")
cuPc2birdview = cuFunctions.get_function("pointcloud_to_birdview")
cuPc_floor_filter = cuFunctions.get_function("get_floor_pointcloud_filter")
cuClean_birdview = cuFunctions.get_function("clean_birdview")


def iDivUp(a, b):
    return a//b + 1

def copy2D_np_to_device(dst, src, type_sz, width, height):
    copy = cuda.Memcpy2D()
    copy.set_src_host(src)
    copy.set_dst_device(dst)
    copy.height = height
    copy.dst_pitch = copy.src_pitch = copy.width_in_bytes = width*type_sz
    copy(aligned=True)

def copy3D_device_to_numpy(dst, src, type_sz, width, height, depth):
    copy = cuda.Memcpy3D()
    copy.set_src_device(src)
    copy.set_dst_host(dst)
    copy.height = height
    copy.depth = depth
    copy.dst_pitch = copy.src_pitch = copy.width_in_bytes = width*type_sz
    copy()

def np_rescale_bbox_xcycwh(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[xc, yc, w, h], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_yx_min_yx_max(bbox_xcycwh: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox_xcycwh: [[y_min, x_min, y_max, x_max], ...]
        @img_size (height, width)
    """
    bbox_xcycwh = np.array(bbox_xcycwh) # Be sure to work with a numpy array
    scale = np.array([img_size[0], img_size[1], img_size[0], img_size[1]])
    bbox_xcycwh_rescaled = bbox_xcycwh * scale
    return bbox_xcycwh_rescaled


def np_rescale_bbox_xy_min_xy_max(bbox: np.array, img_size: tuple):
    """
        Rescale a list of bbox to the image size
        @bbox: [[x_min, y_min, x_max, y_max], ...]
        @img_size (height, width)
    """
    bbox = np.array(bbox) # Be sure to work with a numpy array
    scale = np.array([img_size[1], img_size[0], img_size[1], img_size[0]])
    bbox_rescaled = bbox * scale
    return bbox_rescaled

def np_xcycwh_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [xc, yc, w, h] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (tf.Tensor) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    # convert the bbox from [xc, yc, w, h] to [xmin, ymin, xmax, ymax].
    bbox_xy = np.concatenate([bbox[:, :2] - (bbox[:, 2:] / 2), bbox[:, :2] + (bbox[:, 2:] / 2)], axis=-1)
    return bbox_xy

def np_yx_min_yx_max_to_xy_min_xy_max(bbox: np.array) -> np.array:
    """
    Convert bbox from shape [ymin, xmin, ymax, xmax] to [xmin, ymin, xmax, ymax]
    Args:
        bbox A (np.array) list a bbox (n, 4) with n the number of bbox to convert
    Returns:
        The converted bbox
    """
    return np.concatenate([
        bbox[:,1:2],
        bbox[:,0:1],
        bbox[:,3:4],
        bbox[:,2:3]
    ], axis=-1)

def cuBbox_to_image(
    device_in_img, device_masks, device_resized_masks, device_image_uint8, device_bbox, device_classes,
    host_image_uint8, image_shape,
    bbox_list,  labels, scores, cuda_stream, have_masks=False, class_name=[], input_bbox_format="xcyc",
    display_size = None, device_display_img=None, host_display_img=None, device_plan_dist=None, person_threshold=None, floor_threshold=None
):
    """
    Create an output image with boxes and instance segmentation for only person and floor

    Parameters:
    ------------
        device_*** : DeviceAllocation memory in GPU created by pycuda
        host_*** : pagelocked numpy arrays created by pycuda
        bbox_list : arrays of bbox (n, 4)
        labels : arrays of class id of each box (n, 1)
        scores : confidence of each box (n, 1)
        image_shape : tuple of image shape (H, W, 3)
        display_size : tuple (screen H, screen W, 3)

    Return:
    -------
        image: numpy array in uint8, with color masks and rectangles added
    """


    assert(len(image_shape) == 3)
    
    if input_bbox_format == "xcyc":
        # Convert the bbox format
        bbox_xcycwh = np_rescale_bbox_xcycwh(bbox_list, (image_shape[0], image_shape[1])) 
        bbox_x1y1x2y2 = np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)
    elif input_bbox_format == "yxyx":
        bbox_y1x1y2x2 = np_rescale_bbox_yx_min_yx_max(bbox_list, (image_shape[0], image_shape[1])) 
        bbox_x1y1x2y2 = np_yx_min_yx_max_to_xy_min_xy_max(bbox_y1x1y2x2)
    elif input_bbox_format == "xyxy":
        bbox_x1y1x2y2 = np_rescale_bbox_xy_min_xy_max(bbox_list, (image_shape[0], image_shape[1])) 
    else:
        raise NotImplementedError()

    nbox = bbox_x1y1x2y2.shape[0]
    # Set the labels if not defined
    if labels is None: labels = np.zeros((nbox))

    bbox_area = []
    # Go through each bbox
    for b in range(0, nbox):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2-x1)*(y2-y1))

    # === If mask is not None, create a segmentation image using CUDA
    if have_masks:
        # Copy bbox and labels in GPU
        uint16_labels = labels.astype(np.uint16)
        cuda.memcpy_htod_async(device_classes, uint16_labels, cuda_stream)

        uint16_bbox = bbox_x1y1x2y2.astype(np.uint16)
        cuda.memcpy_htod_async(device_bbox, uint16_bbox, cuda_stream)
        # De-normalize the image in GPU 'cuz it was normalized for model 1
        H, W, C = image_shape
        blockDimY, blockDimX, blockDimZ = 16, 16, 3
        blockDim = (blockDimX, blockDimY, blockDimZ)
        gridDim = (iDivUp(W, blockDimX), iDivUp(H, blockDimY), 1)
        cuDenormalize_image(
            device_in_img, gpu_channel_avg, gpu_channel_std, 
            np.int32(H), np.int32(W), np.int32(C),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        # == Sigmoid masks
        blockDim = (512, 1, 1)
        nb_element = int(H/4*W/4*nbox)
        gridDim = (iDivUp(nb_element, 512), 1, 1)
        cuSigmoid(
            device_masks, np.int32(nb_element),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        # == Resize masks
        blockDimY, blockDimX, blockDimZ = 3, 16, 16
        blockDim = (blockDimX, blockDimY, blockDimZ)
        gridDim = (iDivUp(H, blockDimX), iDivUp(nbox, blockDimY), iDivUp(W, blockDimZ))
        # cuResize_batch_masks(
        cuBilinear_resize_masks(
            device_resized_masks, device_masks, 
            np.int32(image_shape[1]), np.int32(image_shape[0]),
            np.int32(image_shape[1]/4), np.int32(image_shape[0]/4), np.int32(nbox),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )   

        # == Create panoptic image
        blockDimY, blockDimX, blockDimZ = 16, 16, 3
        blockDim = (blockDimX, blockDimY, blockDimZ)
        gridDim = (iDivUp(W, blockDimX), iDivUp(H, blockDimY), 1)
        if person_threshold is None:
            person_threshold = PERSON_THRESHOLD
        if floor_threshold is None:
            floor_threshold = FLOOR_THRESHOLD
        if device_plan_dist is None:
            cuCreate_panoptic(
                device_in_img, device_classes, 
                GPU_CLASS_COLOR_MAP, device_resized_masks, device_bbox,
                np.int32(H), np.int32(W), np.int32(C), np.int32(nbox), np.float32(person_threshold), np.float32(floor_threshold),
                block=blockDim, grid=gridDim, stream=cuda_stream
            )
        else:
            cuCreate_panoptic_combined(
                device_in_img, device_classes, 
                GPU_CLASS_COLOR_MAP, device_resized_masks, device_bbox,
                np.int32(H), np.int32(W), np.int32(C), np.int32(nbox), np.float32(person_threshold), np.float32(floor_threshold),
                device_plan_dist,
                block=blockDim, grid=gridDim, stream=cuda_stream
            )
        cuCvtBGR2RGB(
            device_in_img, np.int32(H), np.int32(W), np.int32(C),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        blockDim = (512, 1, 1)
        gridDim = (iDivUp(host_image_uint8.size, 512), 1, 1)
        cuAstype_uint8(
            device_image_uint8, device_in_img,
            block=blockDim, grid=gridDim, stream=cuda_stream
        )
        # == Resize image to the display size
        if display_size is not None:
            blockDimY, blockDimX, blockDimZ = 3, 16, 16
            blockDim = (blockDimX, blockDimY, blockDimZ)
            gridDim = (iDivUp(display_size[1], blockDimX), iDivUp(display_size[0], blockDimY), 1)
            cuBilinear_resize_image(
                device_display_img, device_image_uint8, 
                np.int32(display_size[1]), np.int32(display_size[0]),
                np.int32(image_shape[1]), np.int32(image_shape[0]), np.int32(image_shape[2]),
                block=blockDim, grid=gridDim, stream=cuda_stream
            )  
            cuda.memcpy_dtoh_async(host_display_img, device_display_img, cuda_stream)
            cuda_stream.synchronize()
            image = host_display_img.reshape(display_size)
        else:
            cuda.memcpy_dtoh_async(host_image_uint8, device_image_uint8, cuda_stream)
            cuda_stream.synchronize()
            image = host_image_uint8.reshape(image_shape)
    else:
        image = host_image_uint8.reshape(image_shape)
        

    # Go through each bbox
    # for b in range(nbox):
    #     # Select the class associated with this bbox
    #     class_id = labels[int(b)]
    #     # Select the bbox to display
    #     # Draw bbox only if it is a person
    #     if class_id == 0:
    #         x1, y1, x2, y2 = bbox_x1y1x2y2[b]
    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #         x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_shape[1], x2), min(image_shape[0], y2)
    #         if len(class_name) == 0:
    #             label_name = "unknow"    
    #         elif scores is not None and len(scores) > 0:
    #             label_name = class_name[int(class_id)]   
    #             label_name = "%s:%.2f" % (label_name, scores[b])
    #         else:
    #             label_name = class_name[int(class_id)]    

    #         class_color = CLASS_COLOR_MAP[int(class_id)].astype(np.uint8)

    #         multiplier = image_shape[0] / 500

    #         cv2.rectangle(image, (x1, y1), (x1 + int(multiplier*15)*len(label_name), y1 + 20), class_color.tolist(), -10)
    #         cv2.putText(image, label_name, (x1+2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * multiplier, (0, 0, 0), 1)
    #         cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)
    return image


def cuBbox_to_image_coco(
    device_in_img, device_masks, device_resized_masks, device_image_uint8, device_bbox, device_classes,
    host_image_uint8, image_shape,
    bbox_list,  labels, scores, cuda_stream, have_masks=False, class_name=[], input_bbox_format="xcyc",
    display_size = None, device_display_img=None, host_display_img=None, threshold=0.5
):
    """
    Create an output image with boxes and instance segmentation.

    Parameters:
    ------------
        device_*** : DeviceAllocation memory in GPU created by pycuda
        host_*** : pagelocked numpy arrays created by pycuda
        bbox_list : arrays of bbox (n, 4)
        labels : arrays of class id of each box (n, 1)
        scores : confidence of each box (n, 1)
        image_shape : tuple of image shape (H, W, 3)
        display_size : tuple (screen H, screen W, 3)

    Return:
    -------
        image: numpy array in uint8, with color masks and rectangles added
    """


    assert(len(image_shape) == 3)
    
    if input_bbox_format == "xcyc":
        # Convert the bbox format
        bbox_xcycwh = np_rescale_bbox_xcycwh(bbox_list, (image_shape[0], image_shape[1])) 
        bbox_x1y1x2y2 = np_xcycwh_to_xy_min_xy_max(bbox_xcycwh)
    elif input_bbox_format == "yxyx":
        bbox_y1x1y2x2 = np_rescale_bbox_yx_min_yx_max(bbox_list, (image_shape[0], image_shape[1])) 
        bbox_x1y1x2y2 = np_yx_min_yx_max_to_xy_min_xy_max(bbox_y1x1y2x2)
    elif input_bbox_format == "xyxy":
        bbox_x1y1x2y2 = np_rescale_bbox_xy_min_xy_max(bbox_list, (image_shape[0], image_shape[1])) 
    else:
        raise NotImplementedError()

    nbox = bbox_x1y1x2y2.shape[0]
    # Set the labels if not defined
    if labels is None: labels = np.zeros((nbox))

    bbox_area = []
    # Go through each bbox
    for b in range(0, nbox):
        x1, y1, x2, y2 = bbox_x1y1x2y2[b]
        bbox_area.append((x2-x1)*(y2-y1))

    # === If mask is not None, create a segmentation image using CUDA
    if have_masks:
        # Copy bbox and labels in GPU
        uint16_labels = labels.astype(np.uint16)
        cuda.memcpy_htod_async(device_classes, uint16_labels, cuda_stream)

        uint16_bbox = bbox_x1y1x2y2.astype(np.uint16)
        cuda.memcpy_htod_async(device_bbox, uint16_bbox, cuda_stream)
        # De-normalize the image in GPU 'cuz it was normalized for model 1
        H, W, C = image_shape
        blockDimY, blockDimX, blockDimZ = 16, 16, 3
        blockDim = (blockDimX, blockDimY, blockDimZ)
        gridDim = (iDivUp(W, blockDimX), iDivUp(H, blockDimY), 1)
        cuDenormalize_image(
            device_in_img, gpu_channel_avg, gpu_channel_std, 
            np.int32(H), np.int32(W), np.int32(C),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        # == Sigmoid masks
        blockDim = (512, 1, 1)
        nb_element = int(H/4*W/4*nbox)
        gridDim = (iDivUp(nb_element, 512), 1, 1)
        cuSigmoid(
            device_masks, np.int32(nb_element),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        # == Resize masks
        blockDimY, blockDimX, blockDimZ = 3, 16, 16
        blockDim = (blockDimX, blockDimY, blockDimZ)
        gridDim = (iDivUp(H, blockDimX), iDivUp(nbox, blockDimY), iDivUp(W, blockDimZ))
        # cuResize_batch_masks(
        cuBilinear_resize_masks(
            device_resized_masks, device_masks, 
            np.int32(image_shape[1]), np.int32(image_shape[0]),
            np.int32(image_shape[1]/4), np.int32(image_shape[0]/4), np.int32(nbox),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )   

        # == Create panoptic image
        blockDimY, blockDimX, blockDimZ = 16, 16, 3
        blockDim = (blockDimX, blockDimY, blockDimZ)
        gridDim = (iDivUp(W, blockDimX), iDivUp(H, blockDimY), 1)
    
        cuCreate_panoptic_coco(
            device_in_img, device_classes, 
            GPU_CLASS_COLOR_MAP, device_resized_masks, device_bbox,
            np.int32(H), np.int32(W), np.int32(C), np.int32(nbox), np.float32(threshold),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        cuCvtBGR2RGB(
            device_in_img, np.int32(H), np.int32(W), np.int32(C),
            block=blockDim, grid=gridDim, stream=cuda_stream
        )

        blockDim = (512, 1, 1)
        gridDim = (iDivUp(host_image_uint8.size, 512), 1, 1)
        cuAstype_uint8(
            device_image_uint8, device_in_img,
            block=blockDim, grid=gridDim, stream=cuda_stream
        )
        print("Checking display_size 1 ", display_size)
        # == Resize image to the display size
        if display_size is not None:
            blockDimY, blockDimX, blockDimZ = 3, 16, 16
            blockDim = (blockDimX, blockDimY, blockDimZ)
            gridDim = (iDivUp(display_size[1], blockDimX), iDivUp(display_size[0], blockDimY), 1)
            print("Checking display_size 2 ", display_size)

            cuBilinear_resize_image(
                device_display_img, device_image_uint8, 
                np.int32(display_size[1]), np.int32(display_size[0]),
                np.int32(image_shape[1]), np.int32(image_shape[0]), np.int32(image_shape[2]),
                block=blockDim, grid=gridDim, stream=cuda_stream
            )  
            print("Checking display_size 3 ", display_size)

            cuda.memcpy_dtoh_async(host_display_img, device_display_img, cuda_stream)

            print("Checking display_size 4 ", display_size)
            cuda_stream.synchronize()
            print("Checking display_size 5 ", display_size)

            image = host_display_img.reshape(display_size)
        else:
            cuda.memcpy_dtoh_async(host_image_uint8, device_image_uint8, cuda_stream)
            cuda_stream.synchronize()
            image = host_image_uint8.reshape(image_shape)
    else:
        image = host_image_uint8.reshape(image_shape)
        

    # # Go through each bbox
    # for b in range(nbox):
    #     # Select the class associated with this bbox
    #     class_id = labels[int(b)]
    #     # Select the bbox to display
    #     x1, y1, x2, y2 = bbox_x1y1x2y2[b]
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    #     x1, y1, x2, y2 = max(0, x1), max(0, y1), min(image_shape[1], x2), min(image_shape[0], y2)
    #     if len(class_name) == 0:
    #         label_name = "unknow"    
    #     elif scores is not None and len(scores) > 0:
    #         label_name = class_name[int(class_id)]   
    #         label_name = "%s:%.2f" % (label_name, scores[b])
    #     else:
    #         label_name = class_name[int(class_id)]    

    #     class_color = CLASS_COLOR_MAP[int(class_id)].astype(np.uint8)

    #     multiplier = image_shape[0] / 500

    #     cv2.rectangle(image, (x1, y1), (x1 + int(multiplier*15)*len(label_name), y1 + 20), class_color.tolist(), -10)
    #     cv2.putText(image, label_name, (x1+2, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * multiplier, (0, 0, 0), 1)
    #     cv2.rectangle(image, (x1, y1), (x2, y2), tuple(class_color.tolist()), 2)
    return image


def cuPreprocess_image(host_image_uint8, device_in_img, device_image_uint8, image_shape):
    H, W, C = image_shape
    # Copy the image in the GPU
    cuda.memcpy_htod(device_image_uint8, host_image_uint8)
    # Convert img from uint8 to float32
    blockDim = (512, 1, 1)
    gridDim = (iDivUp(H*W*C, 512), 1, 1)
    cuUint8_to_float32(
        device_in_img, device_image_uint8, 
        block=blockDim, grid=gridDim
    )
    # Convert color and normalize image
    blockDimY, blockDimX, blockDimZ = 16, 16, 3
    threads_per_block = blockDimX*blockDimY*blockDimZ
    blockDim = (blockDimX, blockDimY, blockDimZ)
    gridDim = (iDivUp(W, blockDimX), iDivUp(H, blockDimY), iDivUp(C, blockDimZ))
    cuCvtBGR2RGB(
        device_in_img, np.int32(H), np.int32(W), np.int32(C),
        block=blockDim, grid=gridDim
    )
    cuNormalize_image(
        device_in_img, gpu_channel_avg, gpu_channel_std, 
        np.int32(H), np.int32(W), np.int32(C),
        block=blockDim, grid=gridDim
    )
    return device_in_img
