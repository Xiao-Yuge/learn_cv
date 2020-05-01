import cv2
import numpy as np


def conv_2d(image, kernel, stride=1):
    """
    二维卷积操作
    :param image: 原图
    :param kernel: 卷积核
    :param stride: 步长
    :return: 卷积结果
    """
    return

def cal_grad(gx, gy):
    """
    计算像素梯度与梯度方向
    :param gx: x方向上边缘梯度
    :param gy: y方向上边缘梯度
    :return: 灰度梯度与梯度方向
    """
    return

def sobel(image):
    """
    使用sobel算子计算灰度梯度与灰度梯度方向
    :param image: 原图
    :return image_grad: 图像灰度梯度
            image_grad_orien: 图像灰度方向
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.uint8)
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.uint8)

    image_border_x = conv_2d(image, Gx)
    image_border_y = conv_2d(image, Gy)
    image_grad, image_grad_orien = cal_grad(image_border_x, image_border_y)
    return image_grad, image_grad_orien

def sobel_border(image, threshold):
    image_grad, image_grad_orien = sobel(image)
    image = np.where(image_grad > threshold, 255, 0)
    image = np.cast(image, np.uint8)
    cv2.imshow("sobel边缘", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()