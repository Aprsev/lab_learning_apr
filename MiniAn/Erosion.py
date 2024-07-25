import cv2
import numpy as np
img=cv2.imread("d:\\Desktop\\test\\he.webp")
def cv_show(img,winname):
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
cv_show(img,"origin")
kernel = np.ones((3, 3), dtype=np.uint8)
dilate1 = cv2.dilate(img, kernel, iterations = 1) # 1:迭代次数，也就是执行几次膨胀操作cv_show(dilate)
cv_show(dilate1,"dilated 1")
kernel = np.ones((8, 8), dtype=np.uint8)
dilate2 = cv2.dilate(img, kernel, iterations = 1)
cv_show(dilate2,"dilated 2")
kernel = np.ones((3, 3), dtype=np.uint8)
erosion1 = cv2.erode(img, kernel, iterations = 1)
ss = np.hstack((img, erosion1))
cv_show(ss,"erosion 1")
kernel = np.ones((3,3), dtype=np.uint8)
erosion2 = cv2.erode(img, kernel, iterations = 2)
ss = np.hstack((img, erosion2))
cv_show(ss,"erosion 2")
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
cv_show(gradient,"gradient")
top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
ss = np.hstack((img, top_hat))
cv_show(ss,"top_hat")
black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
ss = np.hstack((img, black_hat))
cv_show(ss,"black_hat")