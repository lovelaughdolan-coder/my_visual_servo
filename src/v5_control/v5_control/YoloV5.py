import onnx
import onnxruntime as ort
import cv2
import numpy as np
import time
import torch

CLASSES = ['buttton', 'knob_ne', 'knob_nw', 'nut', 'plate1_off', 'plate1_on', 'plate2_off', 'plate2_on']  # 类别


class YoloV5onnx():
    def __init__(self, onnx_path, half=False):
        self.half = half
        self.central_list = []
        onnx_model = onnx.load(onnx_path)
        try:
            onnx.checker.check_model(onnx_model)
        except Exception:
            print("model error！")
        else:
            print("model success!")

        # 创建会话选项对象
        options = ort.SessionOptions()

        # 启用性能分析
        options.enable_profiling = False
        # 创建ort会话
        self.onnx_session = ort.InferenceSession(onnx_path, sess_options=options, providers=[
            "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"])

        # self.onnx_session = ort.InferenceSession(onnx_path, sess_options=options, providers=["CPUExecutionProvider"])
        self.input_name = self.get_input_name()
        self.output_name = self.get_output_name()

        # warm up
        self.warm_up()  # warm up

    def warm_up(self):
        for i in range(3):
            input_numpy = np.empty((1, 3, 640, 640), dtype=np.float16 if self.half else np.float32)
            input_feed = self.get_input_feed(input_numpy)
            pred = self.onnx_session.run(None, input_feed)[0]
            print("model warm up success!")

    def get_input_name(self):
        # 获取输入节点名称
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_output_name(self):
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), stride=32):
        """ 图像缩放填充 """
        shape = im.shape[:2]

        self.ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # Scale ratio

        new_unpad = int(round(shape[1] * self.ratio)), int(round(shape[0]) * self.ratio)  # w h

        self.dw, self.dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # w, h  padding

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(self.dh - 0.1)), int(round(self.dh + 0.1))
        left, right = int(round(self.dw - 0.1)), int(round(self.dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, self.ratio, (self.dw, self.dh)

    def get_input_feed(self, image_numpy):
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy
        return input_feed

    def inference(self, img_path):
        """
        1. cv2读取图像并resize
        2.图像转 BGR2RGB 和 HWC2CHW (yolov5的onnx模型输入为RGB: 1 * 3 * 640 * 640)
        3.图像归一化
        4.CHW 2 NCHW
        5.onnx_session 推理
        """
        self.img = img_path
        self.or_img, ratio, (dw, dh) = self.letterbox(self.img)
        img = self.or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR 2 RGB 和 HWC 2 CHW
        img = img.astype(dtype=np.half if self.half else np.float32)  # 是否半精度推理
        img /= 255.0
        img = img[None]  # 增加批次N
        input_feed = self.get_input_feed(img)
        start_time = time.time()
        pred = self.onnx_session.run(None, input_feed)[0]
        print("模型推理耗时:", time.time() - start_time)

        return pred

    def xywh2xyxy(self, x):
        """ xywh 2 xyxy """
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def iou(self, a_box, b_box, isMin=False):
        """
        1.计算b_box面积
        2.计算交集坐标
        3.计算交集面积
        3.iou计算 根据isMin，进行交集 / 最小面积  或者  交集 / 并集
        """
        # 如果模型使用半精读推理，float16容易精度溢出，因此计算时转为float32进行计算
        if a_box.dtype == "float16" or b_box.dtype == "float16":
            a_box = a_box.astype(np.float32)
            b_box = b_box.astype(np.float32)
        # 计算面积
        a_box_area = (a_box[2] - a_box[0]) * (a_box[3] - a_box[1])
        b_box_area = (b_box[:, 2] - b_box[:, 0]) * (b_box[:, 3] - b_box[:, 1])

        # 找交集
        xx1 = np.maximum(a_box[0], b_box[:, 0])
        yy1 = np.maximum(a_box[1], b_box[:, 1])
        xx2 = np.minimum(a_box[2], b_box[:, 2])
        yy2 = np.minimum(a_box[3], b_box[:, 3])

        # 判断是否有交集
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # 计算交集面积
        inter = w * h

        # 计算iou
        if isMin:
            ious = np.true_divide(inter, np.minimum(a_box_area, b_box_area))
        else:
            ious = np.true_divide(inter, (a_box_area + b_box_area - inter))

        return ious

    def nms(self, dets, thresh):
        """
        非极大值抑制
        1.按照置信度排序，并得到索引
        2.得到最大的置信度为box_a
        3.其余为box_b
        4.使用box_a 和 box_b 进行iou比对
        5.将满足阈值的iou，保存下来
         """
        if dets.shape[0] == 0:
            return np.array([])
        sort_index = dets[:, 4].argsort()[::-1]  # 从大到小排序

        keep = []
        while sort_index.size > 0:
            keep.append(sort_index[0])

            box_a = dets[sort_index[0]]  # 第一个置信度最高的框
            box_b = dets[sort_index[1:]]  # 其余所有框
            iou = self.iou(box_a, box_b)
            idx = np.where(iou <= thresh)[0]
            sort_index = sort_index[idx + 1]
        return keep

    def filter_box(self, out_put, conf_threshold=0.70, iou_threshold=0.5):
        """ NMS """
        out_put = out_put[0]  # [25200, 85]  85: x, y, w, h, conf, classes80

        # filter conf
        conf = out_put[:, 4] > conf_threshold
        box = out_put[conf == True]  # [57, 85]

        # 使用argmax获取类别
        cls_cinf = box[..., 5:]
        cls = [int(np.argmax(cl)) for cl in cls_cinf]
        all_cls = list(set(cls))  # 去重，获取检出的类别
        """
        分别对每个类别进行过滤
        1.将第6列元素替换为类别下标
        2.xywh 2 xyxy
        3.经过非极大值抑制后输出box下标
        4.利用下标去除非极大值抑制后的box
        """
        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []

            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])  # x1 y1 x2 y2 w h score class
            curr_cls_box = np.array(curr_cls_box)  # x1 y1 x2 y2 w h score class
            curr_cls_box = self.xywh2xyxy(curr_cls_box)  # xywh 2 xyxy
            idx = self.nms(curr_cls_box, iou_threshold)
            for k in idx:
                output.append(curr_cls_box[k])
        return output

    def transform_coords(self, boxes, img_shape):
        """
        将letterbox坐标转换回原始图像坐标系
        
        参数:
            boxes: filter_box返回的bbox列表 [x1, y1, x2, y2, conf, cls]
            img_shape: 原始图像形状 (height, width)
            
        返回:
            转换后的bbox列表
        """
        transformed_boxes = []
        h, w = img_shape[:2]
        
        for box in boxes:
            x1 = int((box[0] - self.dw) / self.ratio)
            y1 = int((box[1] - self.dh) / self.ratio)
            x2 = int((box[2] - self.dw) / self.ratio)
            y2 = int((box[3] - self.dh) / self.ratio)
            
            # 边界检查
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            transformed_boxes.append([x1, y1, x2, y2] + list(box[4:]))
        
        return np.array(transformed_boxes)

    def draw(self, outbox):
        img = self.img.copy()
        for info in outbox:
            conf = info[4]  # 置信度
            cls = int(info[5])  # 类别
            x1 = int((info[0] - self.dw) / self.ratio)
            y1 = int((info[1] - self.dh) / self.ratio)
            x2 = int((info[2] - self.dw) / self.ratio)
            y2 = int((info[3] - self.dh) / self.ratio)
            central_point = [str(CLASSES[cls]), (x2 - x1) / 2 + x1, (y1 - y2) / 2 + y2]
            self.central_list.append(central_point)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{CLASSES[cls]}:{conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img




