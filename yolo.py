# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v3 style detection model on image and video
"""

import colorsys
# import os
from timeit import default_timer as timer
import datetime

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

import cv2

# Example of how to use last several frames info next
# Key word: class, deque
class Line:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False

        # x and y values in last frame
        self.x = None
        self.y = None

        # x intercepts for average smoothing
        self.bottom_x = deque(maxlen=frame_num)
        self.top_x = deque(maxlen=frame_num)

        # Record last x intercept
        self.current_bottom_x = None
        self.current_top_x = None

        # Record radius of curvature
        self.radius = None

        # Polynomial coefficients: x = A*y**2 + B*y + C
        self.A = deque(maxlen=frame_num)
        self.B = deque(maxlen=frame_num)
        self.C = deque(maxlen=frame_num)
        self.fit = None
        self.fitx = None
        self.fity = None
        
    def quick_search(self, nonzerox, nonzeroy):
        """
        Assuming in last frame, lane has been detected. Based on last x/y coordinates, quick search current lane.
        """
        x_inds = []
        y_inds = []
        if self.detected:
            win_bottom = 720
            win_top = 630
            while win_top >= 0:
                yval = np.mean([win_top, win_bottom])
                xval = (np.median(self.A)) * yval ** 2 + (np.median(self.B)) * yval + (np.median(self.C))
                x_idx = np.where((((xval - 50) < nonzerox)
                                  & (nonzerox < (xval + 50))
                                  & ((nonzeroy > win_top) & (nonzeroy < win_bottom))))

# Example end
                
class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/coco_classes.txt',
        "tiny_model_path": 'model_data/trained_weights.h5',  #
        "tiny_anchors_path": 'model_data/tiny_yolo_anchors.txt',  #
        "tiny_classes_path": 'model_data/voc_classes.txt',  #
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()
        self.tiny_class_names = self._get_tiny_class()  #
        self.tiny_anchors = self._get_tiny_anchors()  #
        self.tiny_boxes, self.tiny_scores, self.tiny_classes = self.tiny_generate()  #
        self.count_num = 0
        self.last_num = 1
        self.dis_i = 0
        self.test_date = datetime.datetime.now()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_tiny_class(self):
        tiny_classes_path = os.path.expanduser(self.tiny_classes_path)
        with open(tiny_classes_path) as f:
            tiny_class_names = f.readlines()
        tiny_class_names = [c.strip() for c in tiny_class_names]
        return tiny_class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_tiny_anchors(self):
        tiny_anchors_path = os.path.expanduser(self.tiny_anchors_path)
        with open(tiny_anchors_path) as f:
            tiny_anchors = f.readline()
        tiny_anchors = [float(x) for x in tiny_anchors.split(',')]
        return np.array(tiny_anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def tiny_generate(self):
        tiny_model_path = os.path.expanduser(self.tiny_model_path)
        assert tiny_model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        tiny_num_anchors = len(self.tiny_anchors)
        tiny_num_classes = len(self.tiny_class_names)
        is_tiny_version = tiny_num_anchors==6 # default setting
        try:
            self.tiny_yolo_model = load_model(tiny_model_path, compile=False)
        except:
            self.tiny_yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), tiny_num_anchors//2, tiny_num_classes) \
                if is_tiny_version else tiny_yolo_body(Input(shape=(None,None,3)), tiny_num_anchors//3, tiny_num_classes)
            self.tiny_yolo_model.load_weights(self.tiny_model_path) # make sure model, anchors and classes match
        else:
            assert self.tiny_yolo_model.layers[-1].output_shape[-1] == \
                tiny_num_anchors/len(self.yolo_model.output) * (tiny_num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} tiny_model, tiny_anchors, and tiny_classes loaded.'.format(tiny_model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.tiny_class_names), 1., 1.)
                      for x in range(len(self.tiny_class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.tiny_input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.tiny_yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        tiny_boxes, tiny_scores, tiny_classes = yolo_eval(self.tiny_yolo_model.output, self.tiny_anchors,
                len(self.tiny_class_names), self.tiny_input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return tiny_boxes, tiny_scores, tiny_classes

    def detect_big_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print('Start roi detection')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes

    def detect_num(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print('Start roi detection')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.tiny_boxes, self.tiny_scores, self.tiny_classes],
            feed_dict={
                self.tiny_yolo_model.input: image_data,
                self.tiny_input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return out_boxes, out_scores, out_classes

    def num_trans(self, tiny_out_boxes, tiny_out_scores, tiny_out_classes):
        num_pos = []
        num_char = []
        for tiny_i, tiny_c in reversed(list(enumerate(tiny_out_classes))):
            tiny_predicted_class = self.tiny_class_names[tiny_c]
            tiny_box = tiny_out_boxes[tiny_i]
            tiny_score = tiny_out_scores[tiny_i]
            tiny_top, tiny_left, tiny_bottom, tiny_right = tiny_box
            tiny_left_int = np.floor(tiny_left + 0.5).astype('int32')
            num_pos.append(tiny_left_int)
            if tiny_score > 0.25:
                num_char.append(tiny_predicted_class)
            else:
                num_char.append('0')
        return num_pos, num_char

    def num_filter(self, last_num, num_char, num_pos, count_num):
        num_dis = 0
        last_num_10 = last_num // 10
        last_num_01 = last_num % 10
        last_num_list = [str(last_num_10), str(last_num_01)]
        last_num_10_1 = (last_num - 1) // 10
        last_num_01_1 = (last_num - 1) % 10
        last_num_list_1 = [str(last_num_10_1), str(last_num_01_1)]
        if len(num_char) == 2:
            if num_pos[0] < num_pos[1]:
                num_dis = int(num_char[0]) * 10 + int(num_char[1])
            if num_pos[1] < num_pos[0]:
                num_dis = int(num_char[1]) * 10 + int(num_char[0])
        elif 0 < len(num_char) < 2:
            if num_char in last_num_list_1:
                num_dis = last_num - 1
            else:
                num_dis = last_num
        elif len(num_char) == 0:
            num_dis = 0
        else:
            if last_num_list_1 in num_char:
                num_dis = last_num - 1
            else:
                num_dis = last_num

        if num_dis == 1:
            count_num += 1
        else:
            count_num = 0

        if count_num > 35:
            count_num = 0
            num_dis = 0

        # if last_num > 0:
        #     if last_num == num_dis:
        #         if count_num > 31:
        #             num_dis = last_num - 1
        #             count_num = 0
        #         else:
        #             num_dis = last_num
        #     elif last_num < num_dis:
        #         num_dis = last_num
        #         count_num = 0
        #     else:
        #         count_num = 0
        return num_dis, count_num

    def colour_detect(self, roi):
        roi_hsv = cv2.cvtColor(np.asarray(roi), cv2.COLOR_RGB2HSV)

        lower_red1 = np.array([0, 43, 46])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([156, 43, 46])
        upper_red2 = np.array([180, 255, 255])
        lower_green = np.array([35, 43, 46])
        upper_green = np.array([124, 255, 255])
        lower_yellow = np.array([18, 43, 46])
        upper_yellow = np.array([34, 255, 255])

        mask_green = cv2.inRange(roi_hsv, lower_green, upper_green)
        mask_red1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
        mask_red = mask_red1 ^ mask_red2  # XOR
        mask_yellow = cv2.inRange(roi_hsv, lower_yellow, upper_yellow)

        roi_hsv_val = np.sum(roi_hsv)
        mask_green_val = np.sum(mask_green)
        mask_red_val = np.sum(mask_red)
        mask_yellow_val = np.sum(mask_yellow)

        if mask_green_val > (mask_red_val//2 + mask_yellow_val//2) and mask_green_val/roi_hsv_val > 0.045:
            colour = "Green "
            colour_mark = (0, 255, 0)
        elif mask_red_val > (mask_green_val//2 + mask_yellow_val//2) and mask_red_val/roi_hsv_val > 0.045:
            colour = "Red   "
            colour_mark = (255, 0, 0)
        elif mask_yellow_val > (mask_green_val//2 + mask_red_val//2) and mask_yellow_val/roi_hsv_val > 0.04:
            colour = "Yellow"
            colour_mark = (255, 255, 0)
        else:
            colour = "OFF   "
            colour_mark = (190, 190, 190)
        # print(mask_green_val, mask_red_val, mask_yellow_val)
        # print(colour, mask_green_val/roi_hsv_val, mask_red_val/roi_hsv_val, mask_yellow_val/roi_hsv_val)
        return colour, colour_mark

    def detect_image(self, image):
        # raw_image = image.copy()
        w, h = image.width, image.height
        image_left = int(w / 10 * 3)
        image_top = 0  # int(h / 6)
        image_right = w - image_left
        image_bottom = h/4*3  # - image_top
        image_detect_roi = image_left, image_top, image_right, image_bottom
        image_roi = image.crop(image_detect_roi)
        # raw_image_roi = image_roi.copy()
        out_boxes, out_scores, out_classes = self.detect_big_image(image_roi)
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        tl_index = 0
        num_display = []
        colour_display = []
        colour_str = []
        # Start the Draw
        draw = ImageDraw.Draw(image)
        # self.count_num += 1

        class_to_sort = []
        box_to_sort = []
        box_left = []
        score_to_sort = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            top, left, bottom, right = box
            # Take Traffic Light ONLY
            if (predicted_class == 'traffic light') and (score >= 0.50) and ((bottom - top) * (right - left)) / (image_roi.width * image_roi.height) > 0.001 and ((bottom - top) / (right - left)) > 1.5:
                class_to_sort.append(predicted_class)
                box_to_sort.append(box)
                box_left.append(left)
                score_to_sort.append(score)

        # Sort boxes from left to nine
        if len(box_to_sort) > 0:
            ZIP_Pack = zip(box_left, class_to_sort, box_to_sort, score_to_sort)
            ZIP_Pack_new = sorted(ZIP_Pack, reverse=True)
            new_box_left, new_class, new_box, new_score = zip(*ZIP_Pack_new)
            out_boxes, out_scores, out_classes = new_box, new_score, new_class
        else:
            out_boxes, out_scores, out_classes = box_to_sort, score_to_sort, class_to_sort

        #  End of sort

        for i, c in reversed(list(enumerate(out_classes))):
            # predicted_class = self.class_names[c]
            box = out_boxes[i]
            # score = out_scores[i]
            top, left, bottom, right = box
            # Take Traffic Light ONLY
            # if (predicted_class == 'traffic light') and (score >= 0.50):
            #  and ((bottom - top) * (right - left)) / (
            #  image_roi.width * image_roi.height) > 0.0015:
            tl_index = tl_index + 1

            # Take TL ROI
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (top, left), (bottom, right))
            box_roi = left, top, right, bottom
            roi = image_roi.crop(box_roi)
            # End of ROI

            # Colour detection
            colour, colour_mark = self.colour_detect(roi)
            # End of colour detection

            colour_str.append(colour)
            colour_display.append(colour_mark)

            # Num recognition
            if colour == "Green " or colour == "Red   ":
                tiny_out_boxes, tiny_out_scores, tiny_out_classes = self.detect_num(roi)
            else:
                tiny_out_boxes, tiny_out_scores, tiny_out_classes = [], [], []
            num_pos, num_char = self.num_trans(tiny_out_boxes, tiny_out_scores, tiny_out_classes)
            # End of Num recognition

            # Num filter
            num_dis, count_num = self.num_filter(self.last_num, num_char, num_pos, self.count_num)
            self.last_num = num_dis
            self.count_num = count_num
            # End of filter

            num_display.append(num_dis)

            # Show box
            label = '{}'.format(tl_index)
            font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                          size=np.floor(3e-2 * image_roi.size[1] + 0.5).astype('int32'))
            thickness = (image_roi.size[0] + image_roi.size[1]) // 500
            label_size = draw.textsize(label, font)
            if top - label_size[1] >= 0:
                text_origin = np.array([left+image_left, top+image_top - label_size[1]])
            else:
                text_origin = np.array([left+image_left, top+image_top + 1])

            for i in range(thickness):
                draw.rectangle(
                        [left+image_left + i, top+image_top + i, right+image_left - i, bottom+image_top - i],
                        outline=colour_mark)
            draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colour_mark)
            # draw.text(text_origin, label, fill=(128, 128, 128), font=font)
            draw.text(text_origin, label, fill=(255-colour_mark[0], 255-colour_mark[1], 255-colour_mark[2]), font=font)
            # End of show

        # Show info:
        draw.rectangle(
            [0, image.height / 8 * 6, image.width, image.height],
            fill=(0, 0, 0))
        font_cd = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                     size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        for dis_i in range(tl_index):
            draw.text([image_left, image.height / 8 * 6 + dis_i * 40, image_right, image.height],
                      "|No." + str(dis_i+1) + "| " + colour_str[dis_i] + "| Countdown: " + str(num_display[dis_i]) + " s",
                      fill=colour_display[dis_i], font=font_cd)
            draw.text([image_left, image.height / 8 * 6 + 20 + dis_i * 40, image_right, image.height],
                      "------------------------------",
                      fill=colour_display[dis_i], font=font_cd)
        # End of info

        # Show logo and test date
        font_logo = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                     size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

        draw.text([0, image.height / 8 * 6, image_left, image.height],
                  "BOSCH AI Demo:",
                  fill=(255, 255, 255), font=font_logo)
        draw.text([0, image.height / 8 * 6 + 40, image_left, image.height],
                  "Traffic light",
                  fill=(255, 255, 255), font=font_logo)
        draw.text([0, image.height / 8 * 6 + 80, image_left, image.height],
                  "and Countdown",
                  fill=(255, 255, 255), font=font_logo)
        draw.text([0, image.height / 8 * 6 + 120, image_left, image.height],
                  "Recognition",
                  fill=(255, 255, 255), font=font_logo)
        draw.text([image_right, image.height / 8 * 6, image.width, image.height],
                  "Test date:",
                  fill=(255, 255, 255), font=font_cd)
        draw.text([image_right, image.height / 8 * 6 + 40, image.width, image.height],
                  str(self.test_date)[0:19],
                  fill=(255, 255, 255), font=font_cd)
        # End of show

        # Draw ROI (Optional)
        label = 'ROI'
        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(5e-2 * image_roi.size[1] + 0.5).astype('int32'))
        thickness = (image_roi.size[0] + image_roi.size[1]) // 300
        label_size = draw.textsize(label, font)
        text_origin = np.array([image_left, image_top])  # + label_size[1]])
        for i in range(thickness):
            draw.rectangle(
                [image_left + i, image_top + i, image_right - i, image_bottom - i],
                outline=(0, 0, 0))
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=(0, 0, 0))
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        # End of ROI

        del draw
        # End of Draw

        return image

    def close_session(self):
        self.sess.close()


def detect_video(yolo, video_path, output_path=""):
    # vid = cv2.VideoCapture(0)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                       int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame)
        image = yolo.detect_image(image)
        result = np.asarray(image)

        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo.close_session()
