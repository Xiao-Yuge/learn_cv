import cv2
import numpy as np


"""
仿射变换：平移、旋转、缩放、剪裁、反射
如图：
img:https://s2.ax1x.com/2019/05/30/VKWszD.png
平移(translation)、旋转(rotation)：欧式变换（刚体变换）
缩放(scaling)：
    可分为均匀缩放(uniform scaling)与不均匀缩放(non-uniform scaling)
    均匀缩放：图象在每个坐标轴上的缩放系数相同
    不均匀缩放：图象在每个坐标轴上的缩放系数不同
如果缩放系数为负数，则会叠加上反射（reflection）
相似变换：欧式变换+均匀缩放
剪切变换：图象所有点沿某一方向按比例平移

所有仿射变换都可以使用仿射变换矩阵Q来描述：
[[x'], [y']] = Q * [[x], [y]]
Q = [[a, b], [c, d]]
为了涵盖平移，引入齐次坐标，在原有基础上增广一个维度：
[[x'], [y'], [1]] = Q' * [[x], [y], [1]]
Q' = [[a, b, 0], [c, d, 0], [0, 0, 1]]

常用仿射矩阵：
https://img-blog.csdnimg.cn/20200413000334168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY0NzgxOQ==,size_16,color_FFFFFF,t_70
使用仿射变换矩阵进行图象变换：
img:https://s2.ax1x.com/2019/05/29/VuEg5n.png
"""


def cal_with_matrix(img, matrix):
    def interval(x, i):
        mi, ma = i
        if x >= mi and x < ma:
            return True
        return False

    height, width, channel = img.shape
    new_img = np.zeros(img.shape, dtype=np.uint8)
    new_x, new_y = 0, 0
    for x in range(width):
        for y in range(height):
            coor = matrix @ np.int32([[x], [y], [1]])
            new_x, new_y = np.squeeze(coor)
            # if not interval(new_x, (0, width)) or not interval(new_y, (0, height)):
            #     continue
            if new_x < 0 or new_y >= height:
                break
            elif new_x >= width or new_y < 0:
                continue
            new_img[min(width-1, int(round(new_x))), min(height-1, int(round(new_y)))] = img[x, y]
    return new_img


def translation_cv2(img, x_move, y_move):
    height, width, channel = img.shape
    M = np.float32([[1, 0, x_move], [0, 1, y_move]])
    img_out = cv2.warpAffine(img, M, (width, height))
    cv2.imshow("translation", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translation(img, x_move, y_move):
    height, width, channel = img.shape
    new_img = np.zeros((height, width, channel), dtype=np.uint8)
    if x_move > width or y_move > height:
        return
    for x in range(width):
        if x == width - x_move - 1:
            break
        for y in range(height):
            if y == height - y_move - 1:
                break
            new_img[y+y_move, x+x_move] = img[y, x]
    return new_img


def rotation_cv2(img, theta):
    height, width, channel = img.shape
    M = np.float32([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
    img_out = cv2.warpAffine(img, M, (width, height))
    cv2.imshow("rotation", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def rotation(img, theta):
    M = np.float32([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
    img_out = cal_with_matrix(img, M)
    return img_out


def uniform_scaling_cv2(img, ratio):
    height, width, channel = img.shape
    M = np.float32([[ratio, 0, 0], [0, ratio, 0]])
    img_out = cv2.warpAffine(img, M, (width, height))
    cv2.imshow("uniform scaling", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def uniform_scaling(img, ratio):
    M = np.float32([[ratio, 0, 0], [0, ratio, 0]])
    img_out = cal_with_matrix(img, M)
    return img_out


def non_uniform_scaling_cv2(img, x_ratio, y_ratio):
    height, width, channel = img.shape
    M = np.float32([[x_ratio, 0, 0], [0, y_ratio, 0]])
    img_out = cv2.warpAffine(img, M, (width, height))
    cv2.imshow("non-uniform scaling", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def non_uniform_scaling(img, x_ratio, y_ratio):
    M = np.float32([[x_ratio, 0, 0], [0, y_ratio, 0]])
    img_out = cal_with_matrix(img, M)
    # cv2.imshow("non-uniform scaling", img_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_out


def similarity_transformation(img, x_move, y_move, theta, ratio):
    img = translation(img, x_move, y_move)
    img = rotation(img, theta)
    return uniform_scaling(img, ratio)


def shear_mapping_cv2(img, x_theta, y_theta):
    height, width, channel = img.shape
    M = np.float32([[1, np.tan(x_theta), 0], [0, 1, 0]])
    img_out = cv2.warpAffine(img, M, (width, height))
    return img_out


def shear_mapping(img, x_theta, y_theta):
    M = np.float32([[1, np.tan(x_theta), 0], [0, 1, 0]])
    img_out = cal_with_matrix(img, M)
    return img_out


if __name__ == "__main__":
    img = cv2.imread('../lena.jpg')
    img = non_uniform_scaling(img, 2, 10)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
