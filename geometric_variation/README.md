# 图像变换

## 简介
图像的几何变换包括图像的平移、旋转、缩放、剪裁、反射。几何变换可以改变图像的空间位置，但不改变图像的色彩特性。下图直观地表示了几种常用的图像基础变换。
<br>
![](https://s2.ax1x.com/2019/05/30/VKWszD.png)
<p align="center">图1 常用图形变换示例</p>

##1 仿射变换
图像几何变换的一般定义为：<br>

    g(x, y) = I(u, v)
其中，I(u, v)为输入图像，g(x, y)为输出图像，它从u-v坐标系变换为x-y坐标系。<br>

仿射变换是在几何上定义为两个向量空间之间的一个仿射变换或者仿射映射由一个非奇异的线性变换(运用一次函数进行的变换)接上一个平移变换组成。一个仿射变换对应于一个矩阵和一个向量的乘法，而仿射变换的复合对应于普通的矩阵乘法。<br>

所有仿射变换都可以使用仿射变换矩阵Q来描述：


    Q = [[a, b], [c, d]]

图像的仿射变换可以表示为：

    [[x'], [y']] = Q * [[x], [y]]

为了涵盖平移，引入齐次坐标，在原有基础上增广一个维度：

    Q' = [[a, b, 0], [c, d, 0], [0, 0, 1]]

图像仿射变换可表示为：

    [[x'], [y'], [1]] = Q' * [[x], [y], [1]]

常用的仿射变换矩阵：


![](https://img-blog.csdnimg.cn/20200413000334168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MDY0NzgxOQ==,size_16,color_FFFFFF,t_70)
<p align="center">表1 常用仿射变换矩阵</p>

使用仿射变换矩阵进行图像变换：

![](https://s2.ax1x.com/2019/05/29/VuEg5n.png)
<p align="center">图2 使用仿射变换矩阵进行图像变换示例</p>

##2 欧式变换（刚体变换）
###2.1 平移变化（translation）
平移变换就是在变换前后像素的水平垂直坐标发生变化,其变化前后像素关系如图二Translate所示，其中X，Y分别表示水平与垂直方向上的偏移量。<br>
python-opencv实现：

    def translation_cv2(img, x_move, y_move):
    	height, width, channel = img.shape
    	M = np.float32([[1, 0, x_move], [0, 1, y_move]])
    	img_out = cv2.warpAffine(img, M, (width, height))
    	cv2.imshow("translation", img_out)
    	cv2.waitKey(0)
    	cv2.destroyAllWindows()

python实现:

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
###2.2 旋转变化（rotation）
旋转变换是把图像绕某一点以逆时针或顺时针方向旋转一定的角度。<br>
在计算图像旋转后像素的新坐标时，往往是以(0, 0)处像素为原点进行旋转操作，因此如果要以图像中心进行旋转，需要经过几个步骤：

(1) 坐标原点平移到图像中心<br>
(2) 针对新坐标进行旋转操作<br>
(3) 将坐标原点移回原处<br>

图像旋转像素变换关系如图二Rotate所示，x、y坐标计算如表一旋转变化所示。<br>

python-opencv实现：

    def rotation_cv2(img, theta):
	    height, width, channel = img.shape
	    M = np.float32([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
	    img_out = cv2.warpAffine(img, M, (width, height))
	    cv2.imshow("rotation", img_out)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()

python实现：

使用仿射变换矩阵计算新坐标：

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

旋转：

    def rotation(img, theta):
	    M = np.float32([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0]])
	    img_out = cal_with_matrix(img, M)
	    return img_out
##3 图像缩放（scaling）
图像缩放包含均匀缩放（等比缩放）和不均匀缩放（非等比缩放）两种。
###3.1 均匀缩放(uniform scaling)
图像的x和y轴按照相同比例缩放。图像的放大过程中可能出现空白像素，需要使用插值方法添加空白像素的灰度值。其仿射变换矩阵如图二Scale所示，在进行均匀缩放时，W = H。

python-opencv实现：

    def uniform_scaling_cv2(img, ratio):
	    height, width, channel = img.shape
	    M = np.float32([[ratio, 0, 0], [0, ratio, 0]])
	    img_out = cv2.warpAffine(img, M, (width, height))
	    cv2.imshow("uniform scaling", img_out)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()

python实现：

    def uniform_scaling(img, ratio):
	    M = np.float32([[ratio, 0, 0], [0, ratio, 0]])
	    img_out = cal_with_matrix(img, M)
	    return img_out
###3.2 不均匀缩放（non-uniform scaling）
图像的x和y轴缩放比例不同。其仿射变换矩阵如图二Scale所示，W表示x方向上缩放比例，H表示y方向上缩放比例。

python-opencv实现：

    def non_uniform_scaling_cv2(img, x_ratio, y_ratio):
	    height, width, channel = img.shape
	    M = np.float32([[x_ratio, 0, 0], [0, y_ratio, 0]])
	    img_out = cv2.warpAffine(img, M, (width, height))
	    cv2.imshow("non-uniform scaling", img_out)
	    cv2.waitKey(0)
	    cv2.destroyAllWindows()

python实现：

    def non_uniform_scaling(img, x_ratio, y_ratio):
	    M = np.float32([[x_ratio, 0, 0], [0, y_ratio, 0]])
	    img_out = cal_with_matrix(img, M)
	    return img_out
###3.3 反射(reflection)
如果缩放系数为负数，则会叠加上反射（reflection），反射的仿射变换如图二Reflection所示。
##4 相似变化（similarity transformation）
相似变换简单而言就是欧式变换+均匀缩放，在进行叠加变换时可以使用仿射变换矩阵相乘得到叠加变换的仿射变换矩阵进行计算。
##5 剪切变换（shear mapping）
图像延某一方向按比例平移的变换，其仿射变换矩阵如图二shear所示，例如，延x方向上的剪切变换x --> x + tanθ, y保持不变，其中θ表示点延x方向上变换的角度，延y方向上的剪切变换x保持不变，y --> y + tanθ，θ表示点延y方向变换的角度。

python-opencv实现：

    def shear_mapping_cv2(img, x_theta, y_theta):
	    height, width, channel = img.shape
	    M = np.float32([[1, np.tan(x_theta), 0], [0, 1, 0]])
	    img_out = cv2.warpAffine(img, M, (width, height))
	    return img_out

python实现：

    def shear_mapping(img, x_theta, y_theta):
	    M = np.float32([[1, np.tan(x_theta), 0], [0, 1, 0]])
	    img_out = cal_with_matrix(img, M)
	    return img_out

## 待改进
在用python实现不同的变换过程中，在做图像拉伸或放大过程中出现了很多空白像素点，暂时还未用相应的插值方法填充，在后续过程中会陆续优化。
