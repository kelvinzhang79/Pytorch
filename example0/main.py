# 这是一个示例 Python 脚本。
import torch
import numpy as np
import cv2
from torchvision import models
from cyclegan import ResNetGenerator
from cyclegan import ResNetBlock
from PIL import Image
from torchvision import transforms

print(dir(models))

netG = ResNetGenerator()

model_path = './data/p1ch2/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)
netG.eval()
preprocess = transforms.Compose([transforms.Resize(256),
                                 transforms.ToTensor()])
img = Image.open("./data/p1ch2/horse.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)
batch_out = netG(batch_t)
out_t = (batch_out.data.squeeze() + 1.0) / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img

img = cv2.imread('img.jpg', cv2.IMREAD_GRAYSCALE)
print(type(img))
edge_img = cv2.Canny(img, 50, 100)
cv2.imshow('image', edge_img)
k = cv2.waitKey(0)
ret = torch.cuda.is_available()
print(ret)
