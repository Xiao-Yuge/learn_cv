# 图像分割与二值化

##阈值分割基本概念
阈值分割是一种基于区域的图像分割方法，应用于前景与后景灰度值在不同区域的图像的分割（也可以理解为将区域按照像素值的分布区间分成多个类别以达到分离出某些区域的效果）。阈值分割具有简单、计算量小、性能稳定的优点。

##最大类间方差法（大津法）
大津法是一种确定图像二值化分割阈值的方法。通过大津法求得的阈值对图像进行二值化分割后，前景与背景的类间方差最大。大津法不受图像的亮度与对比度影响，它是按照图像的灰度特性，将图像分为前景与背景两部分。图像灰度的方差可以表示图像灰度分布的均匀性，前景与后景灰度的方差越大表示前后景的差别越大，因此按照类间方差最大的方式分割可以得到最佳的前后景分割效果。

大津法适用于需要求图像全局阈值的场景。

优点：简单、计算量小、性能稳定，不受亮度对比度影响。

缺点：对噪声敏感；只能对单一目标分割；前景背景大小比例差距过大时效果不好；对于灰度值渐变时的分割效果很差；

最大类间方差法：

	假设最大类间法将图像像素分为两类：C1,C2
	C1像素的均值为m1，C2像素均值为m2，图像像素的全局均值为mG
	图像像素为C1的概率为p1，为C2的概率为p2
	
	图像像素全局均值mG可以表示为：mG = p1*m1 + p2*m2  (1)
	
	其中p1 + p2 = 1  (2)
	
	根据方差的概念，类间方差可以表示为：σ^2 = p1*(m1 - mG)^2 + p2*(m2 - mG)^2  (3)

	(1),(2)代入(3):σ^2 = p1 * p2 * (m1-m2)^2  (4)

	假设图像有s个像素点，遍历图像的灰度分布，找到一个k使得(4)最大，其中p1 = k/s, p2 = (s-k)/s

分割就是二值化，opencv中的几种分割方式：

	thresh_binary:大于阈值就设为最大值，否则为0

	thresh_binary_inv:大于阈值就设为0，否则为最大值

	thresh_trunc：大于阈值为阈值

	thresh_tozero：小于阈值为0

	thresh_tozero_inv：大于阈值为0
	
##自适应阈值分割

大津法是基于区域的全局阈值分割的方法，对于一些灰度值渐变的图像来说表现效果不好。自适应阈值法是根据图像的不同区域的亮度分布来计算局部阈值，对于图像不同区域可以自适应阈值。

自适应阈值通过计算邻域内灰度的均值、中值、高斯平均加权来确定阈值。如果用局部的均值作为局部的阈值，就是移动平均法。

##大津法代码实现（基于python-opencv）

	import numpy as np
	import cv2
	from collections import Counter
	
	
	def cal_otsu(img):
	    vals = []
	    vals_sum = np.array([0, 0, 0])
	    for h in range(img.shape[0]):
	        for w in range(img.shape[1]):
	            val = img[h, w]
	            vals.append(val)
	            vals_sum += val
	    vals = np.array(vals)
	    vals = list(dict(sorted(Counter(vals[:, i]).items(), key=lambda x:x[0])) for i in range(vals.shape[1]))
	
	    def cal_thresh(dic, vals_sum, pot_sum):
	        v1 = 0
	        s1 = 0
	        max_sigma = 0
	        max_sigma_k = 0
	        for k, v in dic.items():
	            v1 += k*v
	            s1 += v
	            v2 = vals_sum - v1
	            s2 = pot_sum - s1
	            m1 = v1 / s1
	            m2 = v2 / s2
	            p1 = s1 / pot_sum
	            p2 = s2 / pot_sum
	            sigma = p1 * p2 * (m1 - m2)**2
	            if sigma > max_sigma:
	                max_sigma_k = k
	        return max_sigma_k
	
	    thresh = [cal_thresh(vals[i], vals_sum[i], img.shape[0]*img.shape[1]) for i in range(len(vals))]
	    return np.array(thresh)
	
	
	img = cv2.imread("../lena.jpg")
	img_out = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
	# 计算多个通道的最大间类方差
	thresh = cal_otsu(img)
	
	dst, img_out = cv2.threshold(img_out, np.mean(thresh), 255, cv2.THRESH_OTSU)
	
	cv2.imshow("aa", img_out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()