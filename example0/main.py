# 这是一个示例 Python 脚本。
import torch
import numpy as np
import cv2
from torchvision import models
print(dir(models))

img = cv2.imread('img.jpg',cv2.IMREAD_GRAYSCALE)
print(type(img))
edge_img = cv2.Canny(img,50,100)
cv2.imshow('image',edge_img)
k = cv2.waitKey(0)
ret = torch.cuda.is_available()
print(ret)
