import cv2
import numpy as np

def read_image(image_path):
    """读取图像"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def image_standardization(image):
    """图像标准化"""
    mean = np.mean(image)
    std = np.std(image)
    standardized_image = (image - mean) / std
    return standardized_image

def image_normalization(image, a=0, b=1):
    """图像归一化"""
    normalized_image = a + ((image - np.min(image)) * (b - a) / (np.max(image) - np.min(image)))
    return normalized_image

def histogram_equalization(image):
    """直方图均衡化"""
    equ_image = cv2.equalizeHist(image)
    return equ_image

def histogram_specification(source_image, reference_image):
    """直方图规定化"""
    source_hist = cv2.calcHist([source_image], [0], None, [256], [0, 256])
    reference_hist = cv2.calcHist([reference_image], [0], None, [256], [0, 256])
    
    diff = np.zeros(256)
    for i in range(256):
        diff[i] = np.sum(np.abs(source_hist[i] - reference_hist[i]))
    
    corr_val = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if (np.sum(np.abs(source_hist[i] - reference_hist[j])) < diff[i]):
                diff[i] = np.sum(np.abs(source_hist[i] - reference_hist[j]))
                corr_val[i] = j
    
    spec_image = cv2.LUT(source_image, corr_val)
    return spec_image

# 图像路径
image_path = "d:\\Desktop\\test\\he.webp"
reference_image_path = "d:\\Desktop\\test\\de.jpg"

# 读取图像
image = read_image(image_path)
reference_image = read_image(reference_image_path)

# 显示原始图像
cv2.imshow("Original Image", image)
cv2.imshow("Reference Image", reference_image)

# 图像标准化
standardized_image = image_standardization(image)
cv2.imshow("Standardized Image", standardized_image)

# 图像归一化
normalized_image = image_normalization(image)
cv2.imshow("Normalized Image", normalized_image)

# 直方图均衡化
equ_image = histogram_equalization(image)
cv2.imshow("Histogram Equalized Image", equ_image)

# 直方图规定化
spec_image = histogram_specification(image, reference_image)
cv2.imshow("Histogram Specified Image", spec_image)

# 等待按键，然后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()