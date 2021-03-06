# OpenCV框架与图像插值算法
## 简介
&emsp;&emsp;图象基础几何变换包括平移、旋转、缩放、镜像等操作。在这些操作中，输出图象的像素会与输入图象中的像素相对应（否则会出现像素值为空的像素点），但是在缩放操作中，由于缩放比例可能导致输出图象像素点对应在输入图象像素点为小数（计算出的结果在输入图象四个像素点之间），这就需要使用图象插值处理获取对应像素点的灰度值，不同的插值方法有不同的精度，也有不同的失真度。  
<br>
&emsp;&emsp;常用插值算法有：最近邻插值、双线性插值、三次样条插值、立方卷积插值
## 最近邻插值
&emsp;&emsp;在灰度重采样中，输出图像的灰度等于离它位置最近的像素的灰度值，称为最近邻插值法。
<br>
&emsp;&emsp;计算步骤：<br>
&emsp;&emsp;&emsp;&emsp;1、计算输出图象像素点P_O(x_o, y_o)在缩放s倍后对应输入图象的像素点P_I(x_i, y_i)：x_o = x_i/s, y_o = y_i/s
<br>
&emsp;&emsp;&emsp;&emsp;2、如果计算的P_I坐标包含小数，则使用向上或向下取整的方式获取到输入图象中与该像素点最近的像素点。
## 双线性插值
&emsp;&emsp;双线性插值法是利用当前像素的四个相邻像素点灰度值，计算得到当前像素的灰度值。
<br>
&emsp;&emsp;例如P(x, y)相邻的四个像素点Q11(x_1, y_1),Q12(x_1, y_2),Q21(x_2, y_1),Q22(x_2, y_2),X方向上的线性插值为：
<br>

    R1 = (x_2-x)/(x_2-x_1)*f(Q11) + (x-x_1)/(x_2-x_1)*f(Q21)
    R2 = (x_2-x)/(x_2-x_1)*f(Q12) + (x-x_1)/(x_2-x_1)*f(Q22)
<br>
&emsp;&emsp;y方向上的线性插值：

    f(P) = (y_2-y)/(y_2-y_1)*R1 + (y-y_1)/(y_2-y_1)*R2<br>
     	 = ((x_2-x)*f(Q11)*(y_2-y) +
            (x-x_1)*f(Q21)*(y_2-y) +
            (x_2-x)*f(Q12)*(y-y_1) +
            (x-x_1)*f(Q22)*(y-y_1))/((y2-y1)*(x2-x_1))
         = [x_2-x, x-x_1] * [[Q11, Q12], [Q21, Q22]] * [[y_2-y], [y-y_1]] / ((y_2-y_1)*(x_2-x_1))
<div align=center>
<img src='./interpolation/inter-linear.jpeg' title='inter-linear-01' style='max-width:600px'></img>
<img src='./interpolation/inter-linear-02.jpeg' title='inter-linear-02' style='max-width:600px'>
</div>
<br>
## 三次样条插值
&emsp;&emsp;本质是多项式插值，通过三次多项式插值构建三次样条曲线模拟连续像素值变化。
## 立方卷积插值
&emsp;&emsp;立方卷积插值精度介于线性插值与三次样条插值之间，但是由于使用卷积，计算量减少。
## 参考链接
&emsp;&emsp;[插值算法比较分析](http://imgtec.eetrend.com/blog/2019/100044397.html)
<br>
&emsp;&emsp;[三次样条插值](https://blog.csdn.net/YhL_Leo/article/details/47707679)
<br>
&emsp;&emsp;[立方体卷积插值](https://blog.csdn.net/shiyimin1/article/details/80141333)