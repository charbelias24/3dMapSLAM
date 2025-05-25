#!/usr/bin/env python3

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import detr.cuda_functions as cufunc

PANOPTIC_CLASS_NAMES = [
    "person", 'carpet', 'dirt', 'floor-mable', 'floor-other', 'floor-stone',
    'floor-tile', 'floor-wood', 'gravel', 'gournd-other', 'mud', 'pavement', 'platform', 'playingfield',
    'railroad', 'road', 'sand', 'snow', 'background'
]

NB_CLASS = len(PANOPTIC_CLASS_NAMES)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def filter_predictions(pred_boxes, pred_logits, background_id):
    """
    Filter the predictions by removing background detections.
    """
    predicted_scores = np.max(sigmoid(pred_logits), axis=-1)
    predicted_labels = np.argmax(pred_logits, axis=-1)
    # The following is is different if the panoptic model is used
    indices = predicted_labels != background_id  # NB_CLASS - 1
    predicted_scores = predicted_scores[indices]
    predicted_labels = predicted_labels[indices]
    predicted_bbox = pred_boxes[indices]

    return predicted_bbox, predicted_labels, predicted_scores, indices

class DetrPanopticTRT():
    """
    Helper class to run Detr inference from 2 TensorRT engines:
        1. Backbone + Transformer + Box Predictor
        2. Mask Predictor
    """
    def __init__(self, engine1_path, engine2_path, image_shape, stream=None, class_name=PANOPTIC_CLASS_NAMES):
        """
        Parameters:
        -----------
        engine1_path: str, Path to engine 1 (Backbone + Transformer)
        engine2_path: str, Path to engine 2 (Mask Head)
        image_shape: tuple, input shape defined by the model
        stream: pycuda.driver.Stream
        class_name: list[str], class_name used by the model

        """
        self.IMG_SHAPE = image_shape
        self.CLASS_NAME = class_name
        self.NB_CLASS = len(class_name)

        # cuda.init()
        # self.cuda_context = cuda.Device(0).make_context()

        if stream is None:
            self.stream = cuda.Stream()
        else:
            self.stream = stream

        # Read engine 1
        with open(engine1_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            print("Reading engine Detr Core ...")
            self.engine1 = runtime.deserialize_cuda_engine(f.read())
        self.context1 = self.engine1.create_execution_context()
        # Read engine 2
        with open(engine2_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            print("Reading engine FPN ...")
            self.engine2 = runtime.deserialize_cuda_engine(f.read())
        self.context2 = self.engine2.create_execution_context()
        print("Engines loaded")

        # Alocate memories
        self.host_in_img = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(0) ), dtype=np.float32)
        self.host_image_uint8 = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(0)), dtype=np.uint8)
        self.host_out_ft_map_3 = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(1)), dtype=np.float32)
        self.host_out_ft_map_2 = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(2)), dtype=np.float32)
        self.host_out_ft_map_1 = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(3)), dtype=np.float32)
        self.host_out_scr_proj = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(4)), dtype=np.float32)
        self.host_enc = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(5)), dtype=np.float32)
        self.host_dec = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(6)), dtype=np.float32)
        self.host_out_cls = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(7)), dtype=np.float32)
        self.host_out_bbox = cuda.pagelocked_empty(trt.volume(self.context1.get_binding_shape(8)), dtype=np.float32)

        self.device_in_img = cuda.mem_alloc(self.host_in_img.nbytes)
        self.device_image_uint8 = cuda.mem_alloc(self.host_image_uint8.nbytes)
        self.device_out_ft_map_3 = cuda.mem_alloc(self.host_out_ft_map_3.nbytes)
        self.device_out_ft_map_2 = cuda.mem_alloc(self.host_out_ft_map_2.nbytes)
        self.device_out_ft_map_1 = cuda.mem_alloc(self.host_out_ft_map_1.nbytes)
        self.device_out_scr_proj = cuda.mem_alloc(self.host_out_scr_proj.nbytes)
        self.device_out_enc = cuda.mem_alloc(self.host_enc.nbytes)
        self.device_out_dec = cuda.mem_alloc(self.host_dec.nbytes)
        self.device_filtered_dec = cuda.mem_alloc(self.host_dec.nbytes)
        self.device_out_cls = cuda.mem_alloc(self.host_out_cls.nbytes)
        self.device_out_bbox = cuda.mem_alloc(self.host_out_bbox.nbytes)

        # Allocate memory for engine 2
        self.Nslot_max = 100 
        self.device_masks = cuda.mem_alloc(int(self.Nslot_max * self.IMG_SHAPE[0]/4 * self.IMG_SHAPE[1]/4 * 4)) # (n slots*94*168) in np.float32
        self.device_bbox_uint16 = cuda.mem_alloc(self.Nslot_max*4*2) # (H*W*type_size_in_byte) in uint16
        self.device_classes_uint16 = cuda.mem_alloc(self.Nslot_max*1*2) # (100, 1) in uint16
        self.device_resized_masks = cuda.mem_alloc(self.Nslot_max*self.IMG_SHAPE[0]*self.IMG_SHAPE[1]*4) # float32 = 4 bytes

    def _load_input_image(self, input_image):
        """
        Preprocess image and transfer it to GPU
        """
        assert self.IMG_SHAPE == input_image.shape, "Input frame of shape {} does not match {}".format(input_image.shape, self.IMG_SHAPE)
        np.copyto(self.host_image_uint8, input_image.ravel())
        self.device_in_img = cufunc.cuPreprocess_image(
            self.host_image_uint8,
            self.device_in_img,
            self.device_image_uint8,          
            self.IMG_SHAPE
        )
    def _execute_engine_1(self):
        self.context1.execute(
            bindings=[
                int(self.device_in_img), 
                int(self.device_out_ft_map_3),
                int(self.device_out_ft_map_2),
                int(self.device_out_ft_map_1),
                int(self.device_out_scr_proj),
                int(self.device_out_enc),
                int(self.device_out_dec),
                int(self.device_out_cls), 
                int(self.device_out_bbox)]
        )

    def _execute_async_engine_1(self):
        self.context1.execute_async_v2(
            bindings=[
                int(self.device_in_img), 
                int(self.device_out_ft_map_3),
                int(self.device_out_ft_map_2),
                int(self.device_out_ft_map_1),
                int(self.device_out_scr_proj),
                int(self.device_out_enc),
                int(self.device_out_dec),
                int(self.device_out_cls), 
                int(self.device_out_bbox)],
            stream_handle=self.stream.handle
        )
        

    def _update_engine1_predictions_from_gpu(self):
        # Transfer predictions back from the GPU
        cuda.memcpy_dtoh_async(self.host_out_cls, self.device_out_cls, self.stream)
        cuda.memcpy_dtoh_async(self.host_out_bbox, self.device_out_bbox, self.stream)
        cuda.memcpy_dtoh_async(self.host_dec, self.device_out_dec, self.stream)
        self.stream.synchronize()
    
    def _update_filtered_predictions(self):
        self.predicted_bbox, self.predicted_labels, self.predicted_scores, self.filter = filter_predictions(
            self.host_out_bbox.reshape((100, 4)), 
            self.host_out_cls.reshape(100, self.NB_CLASS), 
            self.NB_CLASS - 1
        )

    # TODO: sort slot by scores, then the engine can use the same pointer for decoded slots
    def _update_filtered_decoded_slots(self):
        self.nbox = np.sum(self.filter)
        if self.nbox > 0:
            filtered_dec = self.host_dec.reshape((1, 100, 256))[:, self.filter, :].copy()
            cuda.memcpy_htod(self.device_filtered_dec, filtered_dec)
            self.have_masks = True
        else:
            self.have_masks = False

    def _execute_engine_2(self):
        if self.have_masks:
            self.context2.set_binding_shape(5, (1, self.nbox, 256) )
            self.context2.execute_v2(
                bindings=[
                    int(self.device_out_scr_proj), 
                    int(self.device_out_ft_map_3),
                    int(self.device_out_ft_map_2),
                    int(self.device_out_ft_map_1),
                    int(self.device_out_enc),
                    int(self.device_filtered_dec),
                    int(self.device_masks)]
            )

    def _execute_async_engine_2(self):
        if self.have_masks:
            self.context2.set_binding_shape(5, (1, self.nbox, 256) )
            self.context2.execute_async_v2(
                bindings=[
                    int(self.device_out_scr_proj), 
                    int(self.device_out_ft_map_3),
                    int(self.device_out_ft_map_2),
                    int(self.device_out_ft_map_1),
                    int(self.device_out_enc),
                    int(self.device_filtered_dec),
                    int(self.device_masks)],
                stream_handle=self.stream.handle
            )

    def _get_output_masks(self):
        # binding  6 Identity:0 (-1, 94, 168, 1) False output mask
        # return np array of masks (94, 168, -1)
        host_mask = np.zeros((self.nbox, int(self.IMG_SHAPE[0]/4), int(self.IMG_SHAPE[1]/4)), dtype=np.float32)
        if self.nbox > 0:
            cuda.memcpy_dtoh(host_mask, self.device_masks)
        host_mask = np.transpose(host_mask, (1, 2, 0)).copy()
        return host_mask

    def get_all_output(self):
        return self.predicted_bbox, self.predicted_labels, self.predicted_scores, self._get_output_masks()

    def execute(self, input_image):
        # self.cuda_context.push()
        self._load_input_image(input_image)
        self._execute_async_engine_1()
        self._update_engine1_predictions_from_gpu()
        self._update_filtered_predictions()
        self._update_filtered_decoded_slots()
        self._execute_async_engine_2()
        # self.cuda_context.pop()
        return self.get_all_output()

    def get_post_processed_image(self):
        """
        Return an output image with boxes and instance segmentation.
        """
        return cufunc.cuBbox_to_image_coco(
            self.device_in_img, self.device_masks, self.device_resized_masks, self.device_image_uint8, self.device_bbox_uint16, self.device_classes_uint16,
            self.host_image_uint8, self.IMG_SHAPE,
            self.predicted_bbox, self.predicted_labels, self.predicted_scores, self.stream, self.have_masks, self.CLASS_NAME, "xcyc"
        )

    # def destroy(self):
        # self.cuda_context.pop()

    