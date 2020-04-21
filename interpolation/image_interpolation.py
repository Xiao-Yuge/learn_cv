import cv2
import numpy as np


"""
图象基础几何变换包括平移、旋转、缩放、镜像等操作。在这些操作中，输出图象的像素会与输入图象中的像素相对应（否则会出现像素值为空的像素点），
但是在缩放操作中，由于缩放比例可能导致输出图象像素点对应在输入图象像素点为小数（计算出的结果在输入图象四个像素点之间），这就需要使用图象插值处理
获取对应像素点的灰度值，不同的插值方法有不同的精度，也有不同的失真度。

常用插值算法有：最近邻插值、双线性插值、三次样条插值、立方卷积插值

三次样条插值：本质是多项式插值，通过三次多项式插值构建三次样条曲线模拟连续像素值变化。

立方卷积插值：立方卷积插值精度介于线性插值与三次样条插值之间，但是由于使用卷积，计算量减少。

参考链接：
http://imgtec.eetrend.com/blog/2019/100044397.html
三次样条插值：https://blog.csdn.net/YhL_Leo/article/details/47707679
立方卷积插值：https://blog.csdn.net/shiyimin1/article/details/80141333
"""


# 最近邻插值
def nearest_interpolation(path="../lena.jpg", out_ratio=(1, 3)):
    """
    最近邻插值：
    1、计算输出图象像素点P_O(x_o, y_o)在缩放s倍后对应输入图象的像素点P_I(x_i, y_i)：
        x_o = x_i/s, y_o = y_i/s
    2、如果计算的P_I坐标包含小数，则使用向上或向下取整的方式获取到输入图象中与该像素点最近的像素点。
    :param path: 输入图象
    :param out_ratio: 缩放比例(height_ratio, width_ratio)
    """
    img_input = cv2.imread(path)
    img_input_shape = img_input.shape
    scaled_height = int(img_input_shape[0]*out_ratio[0])
    scaled_width = int(img_input_shape[1]*out_ratio[1])
    channel_nums = img_input_shape[2]
    img_output = np.zeros((scaled_height, scaled_width, channel_nums))
    for x_o in range(scaled_height):
        x_i = round(x_o / out_ratio[0])
        x_i = x_i if x_i < img_input_shape[0] else x_i-1
        for y_o in range(scaled_width):
            y_i = round(y_o / out_ratio[1])
            y_i = y_i if y_i < img_input_shape[1] else y_i-1
            img_output[x_o, y_o] = img_input[x_i, y_i]
    cv2.imshow("lena-nearest-interpolation", img_output.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inter_nearest_cv2(path="../lena.jpg", ratio=(1, 3)):
    """
    使用python-opencv进行图象插值
    cv2.resize(InputArray src, OutputArray dst, Size, fx, fy, interpolation)
    参数：
        InputArray src	输入图片
        OutputArray dst	输出图片
        Size	输出图片尺寸
        fx, fy	沿x轴，y轴的缩放系数
        interpolation	插入方式：
            INTER_NEAREST   最近邻插值
            INTER_LINEAR    双线性插值（默认设置）
            INTER_AREA  使用像素区域关系进行重采样。
            INTER_CUBIC 4x4像素邻域的双三次插值
            INTER_LANCZOS4  8x8像素邻域的Lanczos插值
    """
    input_img = cv2.imread(path)
    output_img = cv2.resize(input_img, dsize=(0, 0), fx=ratio[1], fy=ratio[0], interpolation=cv2.INTER_NEAREST)
    cv2.imshow("lena-inter-nearest", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 双线性插值
def linear_interpolation(path="../lena.jpg", out_ratio=(2, 2)):
    """
    双线性插值是在最近邻插值基础上的改进：在两个方向上使用线性插值计算像素点灰度（而不是直接拿输入图象像素点灰度值）
    1、计算得到输出图象在输入图像上对应的像素点P(x, y)，其周围像素点Q11(x_1, y_1), Q12(x_1, y_2), Q21(x_2, y_1), Q22(x_2, y_2)
        计算灰度函数f(x)
    2、计算x方向上的线性插值：
        R1 = (x_2-x)/(x_2-x_1)*f(Q11) + (x-x_1)/(x_2-x_1)*f(Q21)
        R2 = (x_2-x)/(x_2-x_1)*f(Q12) + (x-x_1)/(x_2-x_1)*f(Q22)
    3、计算y方向上的线性插值：
        f(P) = (y_2-y)/(y_2-y_1)*R1 + (y-y_1)/(y_2-y_1)*R2
             = ((x_2-x)*f(Q11)*(y_2-y) +
               (x-x_1)*f(Q21)*(y_2-y) +
               (x_2-x)*f(Q12)*(y-y_1) +
               (x-x_1)*f(Q22)*(y-y_1))/((y2-y1)*(x2-x_1))
            = [x_2-x, x-x_1] * [[Q11, Q12], [Q21, Q22]] * [[y_2-y], [y-y_1]] / ((y_2-y_1)*(x_2-x_1))
    """
    def linear_inter(img_input, coord):
        x_1, y_1 = int(coord[0]), int(coord[1])
        x_2 = x_1 + 1 if x_1 < img_input.shape[1]-1 else x_1
        y_2 = y_1 + 1 if y_1 < img_input.shape[0]-1 else y_1
        X = np.array([x_2-coord[0], coord[0]-x_1], dtype=np.float)
        Q = np.array([[img_input[x_1, y_1], img_input[x_1, y_2]],
                      [img_input[x_2, y_1], img_input[x_2, y_2]]], dtype=np.float)
        # Y = np.array([[y_2-coord[1]], [coord[1]-y_1]], dtype=np.float)
        # temp = np.matmul(X, Q)
        # if not (y_2-y_1)*(x_2-x_1):
        #     return np.squeeze(np.array([np.matmul(temp[:, i], Y) for i in range(temp.shape[-1])]))
        # return np.squeeze(np.array([np.matmul(temp[:, i], Y) for i in range(temp.shape[-1])])/((y_2-y_1)*(x_2-x_1)))
        Y = np.array([y_2-coord[1], coord[1]-y_1], dtype=np.float)
        return np.matmul(Y, np.matmul(X, Q))/((y_2-y_1)*(x_2-x_1))

    img_input = cv2.imread(path)
    img_input_shape = img_input.shape
    scaled_height = int(img_input_shape[0] * out_ratio[0])
    scaled_width = int(img_input_shape[1] * out_ratio[1])
    channel_nums = img_input_shape[2]
    img_output = np.zeros((scaled_height, scaled_width, channel_nums))
    for x_o in range(scaled_height):
        x_i = x_o / out_ratio[0]
        for y_o in range(scaled_width):
            y_i = y_o / out_ratio[1]
            img_output[x_o, y_o] = linear_inter(img_input, (x_i, y_i)).astype(np.uint8)
    cv2.imshow("lena-linear-interpolation", img_output.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inter_linear(path="../lena.jpg", ratio=(1, 3)):
    input_img = cv2.imread(path)
    output_img = cv2.resize(input_img, dsize=(0, 0), fx=ratio[1], fy=ratio[0], interpolation=cv2.INTER_LINEAR)
    cv2.imshow("lena-inter-linear", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cubic_convolution():
    """
    立方卷积插值
    :return:
    """
    pass


if __name__ == "__main__":
    linear_interpolation()
