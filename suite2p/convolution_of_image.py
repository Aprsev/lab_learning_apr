import cv2
import numpy as np
 
# 读取图像
image = cv2.imread('D:\\Downloads\\Blue White Motivational Modern Aesthetic Sea Desktop Wallpaper.png')

cv2.imshow('Original Image', image)

# 定义卷积核 （kernel = np.ones((3, 3), np.float32) / 9，平均滤波器核）
kernel = np.array([[-2, -1, 0],
                   [-1,1, 1],
                   [0, 1, 2]])

#kernel = np.ones((9,9),np.float32)/81
output=image 
# 进行卷积操作
for i in range(1):
    output = cv2.filter2D(output,-1, kernel)
 
# 显示原始图像和卷积结果

cv2.imshow('Convolution Result', output)
cv2.waitKey(0)
cv2.destroyAllWindows()