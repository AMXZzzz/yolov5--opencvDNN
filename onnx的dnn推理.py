# ---------------------
# bilibilli: 随风而息
# Time：2022/4/25 21:55
# ---------------------

'''
此py仅仅实现在cpu机器上部署运行,在GPU机器上运行没办法用显卡跑
'''

import time
import cv2
import mss
import numpy as np
from numpy import array

class yolov5():
    def __init__(self, onnx_path, confThreshold=0.25, nmsThreshold=0.45):
        self.classes = ['head', '']  # 标签池,对应显示的标签,
        self.colors = [np.random.randint(0, 255, size=3).tolist() for _ in range(len(self.classes))]
        num_classes = len(self.classes)
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.nl = len(self.anchors)
        self.na = len(self.anchors[0]) // 2
        self.no = num_classes + 5
        self.stride = np.array([8., 16., 32.])
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNetFromONNX(onnx_path)

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    #  1ms
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        #  在满足跨步约束的同时调整图像大小和填充图像
        shape = im.shape[:2]  # 计算当前形状[高度，宽度]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)  # 图片高宽一集合赋值给new

        # 计算图片尺寸的比例
        r = min(new_shape[0] / shape[0],
                new_shape[1] / shape[1])  # 找出图片的宽高(高宽)最小的比例，【0】是高，【1】是宽
        if not scaleup:  # 只缩小，不放大（为了更好的 val mAP），默认跳过
            r = min(r, 1.0)  # 若有大于1的则用1比例，若有小于1的则选最小，更新r

        # 计算填充边缘
        ratio = r, r  # 高宽比，用上面计算的最小r作为宽高比
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # round：四舍五入，，new_unpad(缩放后的尺寸)，注意高宽已经变成宽高
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # 宽高所需要填充的像素，高为361，填充279至640。宽为640，不填充。计算方式是宽减去高和高减去宽.
        if auto:  # 过小矩形，一般为False
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh填充
        elif scaleFill:  # 缩放，一般为False
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽高比

        dw /= 2  # 将填充分为 2 条边，即在两边填充，
        dh /= 2

        if shape[::-1] != new_unpad:  # 缩放后的（640，361），shape(640,640)
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)  # 图片im，绝对裁剪，cv2.INTER_LINEAR：双线性插值（默认设置）填充
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # round：四舍五入，减去0.1和加上0.1进行平衡，防止出现填充超出640范围
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 左右填充，left:0    right:0
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        # 添加边框，像一个相框一样的东西，src ： 输入的图片，top, bottom, left, right ：相应方向上的边框宽度，
        # borderType：定义要添加边框的类型, value：如果borderType为cv2.BORDER_CONSTANT时需要填充的常数值。

        return im, ratio, (dw, dh)  # 返回预处理好的图片，以及他的宽高比（注意宽高的顺序），和一组除2的集合（320，139.5），同时跳转至detect

    def box_area(self, boxes: array):
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    def box_iou(self, box1: array, box2: array):
        """
        :param box1（最终候选框）: [N, 4]  # 最大分数框
        :param box2（最终候选框）: [M, 4]  # 剩余的框
        :return: [N, M]
        """
        area1 = self.box_area(box1)  # N    # area1：[675.19532383]
        area2 = self.box_area(box2)
        # 两个数组各维度大小 从后往前对比一致， 或者有一维度值为1；
        lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
        rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
        wh = rb - lt
        wh = np.maximum(0, wh)  # [N, M, 2]  # X 与 Y 逐位比较取其大者；
        inter = wh[:, :, 0] * wh[:, :, 1]
        iou = inter / (area1[:, np.newaxis] + area2 - inter)
        return iou  # NxM

    def numpy_nms(self, boxes: array, scores: array, iou_threshold: float):  # IOU处理

        idxs = scores.argsort()  # 按概率分数进行降序排列索引 [N]    # idxs：[ 79  59  84  2...]
        keep = []
        while idxs.size > 0:  # 统计数组中元素的个数
            max_score_index = idxs[-1]  # 计算idxs的个数，即多少个候选框   max_score_index：64
            max_score_box = boxes[max_score_index][None, :]  # 候选框中最大分数的方框
            keep.append(max_score_index)  # 将选好的候选框放入列表

            if idxs.size == 1:  # 当只有一个候选框是退出
                break
            idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
            other_boxes = boxes[idxs]
            ious = self.box_iou(max_score_box, other_boxes)  # 跳转至IOU计算模块，分数最大的框和其余框比较 1XM
            idxs = idxs[ious[0] <= iou_threshold]

        keep = np.array(keep)
        return keep

    def xywh2xyxy(self, x):  # 锚框坐标转换
        #  将x的4个锚框从 [x, y, w, h] 转换为 [x1, y1, x2, y2] 其中 xy1=top-left, xy2=bottom-right ，即将原来的中心点和宽高转换成，左上角的点和宽高
        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y = np.copy(x)  # 拷一份原来的中心点
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y   计算原点的y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y  # y 是转换后的原点+宽高

    def non_max_suppression(self, prediction, conf_thres=0.25, agnostic=False):  # 非最大抑制 ,conf_thres置信度
        # candidates:候选框       获取置信度，prediction为所有的预测结果. shape(1, 25200, 21),batch为1，25200个预测结果，21 = x,y,w,h,c + class个数
        xc = prediction[..., 4] > conf_thres
        # 参数定义
        min_wh, max_wh = 2, 4096  # 设置最小和最大锚框宽度和高度的上限，max_wh:4096   min_wh:2
        max_nms = 3000  # 设置锚框的上限, 原max_nms:30000
        output = [np.zeros((0, 6))] * prediction.shape[0]  # 返回来一个给定形状和类型的用0填充的数组

        for xi, x in enumerate(prediction):  # 遍历prediction图像索引xi，图像推理结果x 。列表enumerate函数表示列出
            # 判断置信度
            x = x[xc[xi]]  # 获取confidence大于conf_thres的结果，即获取大于conf_thres的置信度
            if not x.shape[0]:  # 如果推理结果为空，则跳过本次遍历，开始下一轮遍历
                continue
            # Compute conf 计算配置,计算xywh
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # 框的中心点转换成x1y1,宽和高转换成x2,y2
            box = self.xywh2xyxy(x[:, :4])  # 跳转至锚框转换函数，传入推理结果x的锚框索引
            # 检测方框矩阵
            conf = np.max(x[:, 5:], axis=1)  # 获取类别最高的置信度，
            j = np.argmax(x[:, 5:], axis=1)  # 获取下标 ，即获取索引
            # 转为array：  x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres],
            re = np.array(conf.reshape(-1) > conf_thres)
            # 转为维度
            conf = conf.reshape(-1, 1)
            j = j.reshape(-1, 1)
            # numpy的拼接
            x = np.concatenate((box, conf, j), axis=1)[re]
            # 检查shape
            n = x.shape[0]  # 获取锚框的个数  n:110
            if not n:  # 若没有锚框，跳过本次遍历
                continue
            elif n > max_nms:  # 过多的锚框，即个数大于max_nms = 30000，按置信度排序
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # 按置信度排序
            # 批处理非极大抑制
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = self.numpy_nms(boxes, scores, self.nmsThreshold)  # numpy_nms 跳转NMS处理函数numpy_nms
            output[xi] = x[i]  # 将IOU对应推理结果的索引赋值给output
        return output

    # 230 ms
    def detect(self, srcimg):  # 接受图片
        t0 = time.time()
        # ==== 6ms ==== # 预处理
        # im = srcimg.copy()  # 拷贝一张图片给新对象im
        im, ratio, wh = self.letterbox(srcimg, self.inpWidth, auto=False)  # 预处理，跳转至letterbox， 输入原图片
        # 设置网络的输入
        blob = cv2.dnn.blobFromImage(im, 1 / 255.0, swapRB=True,
                                     crop=False)  # 对图像进行预处理，包括减均值，比例缩放，裁剪，交换通道等，返回一个4通道的blob(blob可以简单理解为一个N维的数组，用于神经网络的输入)，image:输入图像
        # scalefactor:图像各通道数值的缩放比例，swapRB:交换RB通道，默认为False.(cv2.imread读取的是彩图是bgr通道)，crop:图像裁剪,默认为False.当值为True时，先按比例缩放，然后从中心裁剪成size尺寸

        self.net.setInput(blob)  # net这个类中定义了创建和操作网络的方法，详情查看cv的setinput接口档案

        # ===== 195ms =====     # 推理部分
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # NMS 非最大抑制,即过滤过多的框        # 0.1ms
        pred = self.non_max_suppression(outs, self.confThreshold, agnostic=False)  # 非最大抑制,即过滤过多的框，non_max_suppression

        # 画框
        for i in pred[0]:  # 遍历所有的框   i:框的索引
            left = int((i[0] - wh[0]) / ratio[0])  # 原点的x，即宽      #ratio : 宽高比（注意顺序）
            top = int((i[1] - wh[1]) / ratio[1])  # 计算原点的y
            width = int((i[2] - wh[0]) / ratio[0])
            height = int((i[3] - wh[1]) / ratio[1])

            # 画框方式1
            color = (0, 255, 0)  # RGB     框的颜色
            cv2.rectangle(srcimg, (left, top), (width, height), color, thickness=2)  # 3代表方框线条粗细

            # # 画框方式2
            # cv2.rectangle(srcimg, (int(left), int(top)), (int(width), int(height)), colors(classId, True), thickness=2,
            #               lineType=cv2.LINE_AA)  # 作用是在图像上绘制一个简单的矩形, 返回值：它返回一个图像。cv2.rectangle(image, start_point, end_point, color, thickness)
            # # start_point：它是矩形的起始坐标。坐标表示为两个值的元组，即(X坐标值，Y坐标值),
            # # end_point：它是矩形的结束坐标。坐标表示为两个值的元组，即(X坐标值ÿ坐标值)。
            # # color:它是要绘制的矩形的边界线的颜色。对于BGR，我们通过一个元组。例如：(255，0，0)为蓝色。
            # # thickness:它是矩形边框线的粗细像素。厚度-1像素将以指定的颜色填充矩形形状。
            # # lineType ：线的类型。
            # #
            # label = '%.2f' % conf  # 写置信度  babel：'head':0.97
            # label = '%s:%s' % (self.classes[int(classId)], label)  # 写标签，# self.classes = ['head','cat']

            # 绘制置信度,标签
            conf = i[4]  # 方框对应的置信度索引，用来写置信度
            classId = i[5]  # 方框对应的标签索引,用来匹配标签池
            label = '%s' % self.classes[int(classId)]  # 在标签池选择对应的标签
            # 在边界框顶部显示标签
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, labelSize[1])  # y的最高点
            cv2.putText(srcimg, label, (int(left), int(top - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                        thickness=1, lineType=cv2.LINE_AA)
            # cv2.putText(img,'there 0 error(s):',(50,150),cv2.FONT_HERSHEY_COMPLEX,6,(0,0,255),25)
            # 各参数依次是：照片/添加的文字/左上角坐标（xy）/字体/字体大小/颜色/字体粗细/线的类型

        cv2.putText(srcimg, "FPS:{:.1f}".format(1. / (time.time() - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 235), 4)
        t0 = time.time

        return srcimg  # 返回处理好的图片

if __name__ == '__main__':
    scr = mss.mss()  # 实例化截图
    onnx_path = 'CF.onnx'  # 模型路径
    model = yolov5(onnx_path=onnx_path)  # 传入onnx模型
    scr_width, scr_height = 1920, 1080  # 定义显示器宽高
    resize_win_width, resize_win_height = scr_width // 3, scr_height // 3  # 显示大小
    game_left, game_tap, game_x, gane_y = scr_width // 3, scr_height // 3, scr_width // 3, scr_height // 3  # 截图大小
    monitor = {
        'left': game_left,  # 起始点
        'top': game_tap,  # 起始点
        'width': game_x,  # 长度
        'height': gane_y  # 高度
    }

    cv2.namedWindow('onnx_img', cv2.WINDOW_NORMAL)  # 步骤5.1 指定窗口类型
    while True:
        # 截图加转换 12ms
        start = time.time()
        if not cv2.getWindowProperty('onnx_img', cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            break
        img = scr.grab(monitor)  # 截图
        img = np.array(img)  # 转为numpy类型
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)  # 截图的图片是4个通道[1,4,640,640]，网络输入需要3通道[1,3,640,640]，所有要将4通道转为3通道。

        img = model.detect(img)  # 将图片传入推理模型，接收返回的图片      # ==== 205ms =====

        cv2.imshow('onnx_img', img)  # 展示图片
        cv2.resizeWindow('onnx_img', resize_win_width, resize_win_height)  # 裁剪显示窗口
        end = time.time()

        print(round((end - start) * 1000, 2), 'ms')
        cv2.waitKey(1)  # 显示窗口, cv2.waitKey必须在cv2.imshow后调用

        # 下一步，优化延迟速度,考虑用微软的onnxruntime来做,相比opencv的dnn快上50ms左右
