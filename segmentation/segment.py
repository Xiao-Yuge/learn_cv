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