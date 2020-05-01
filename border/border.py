# coding:utf8
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
    k_height, k_width = kernel.shape
    auto_pad = (int(k_height/2), int(k_width/2))
    image = np.pad(image, auto_pad)
    image_out = image.copy()
    height, width = image.shape
    # 可以改为滑动窗口算法
    for y in range(0, height-k_height, stride):
        for x in range(0, width-k_width, stride):
            temp = image[y:y+k_width, x:x+k_height]
            image_out[y][x] = np.sum(temp*kernel)
    return image_out[auto_pad[0]:-auto_pad[0], auto_pad[1]:-auto_pad[1]]

def cal_grad(gx, gy):
    """
    计算像素梯度与梯度方向
    :param gx: x方向上边缘梯度
    :param gy: y方向上边缘梯度
    :return: 灰度梯度与梯度方向
    """
    grad = np.sqrt(np.square(gx) + np.square(gy))
    grad_orien = np.arctan((gy+.1)/(gx+.1))
    return grad, grad_orien

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
    image = image.astype(np.uint8)
    cv2.imshow("sobel", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image = cv2.imread(r"../lena.jpg")
    sobel_border(image, 15)
