import cv2
import numpy as np

net = cv2.dnn.readNetFromTorch('Week2/models/instance_norm/mosaic.t7')
img = cv2.imread('Week2/imgs/03.jpg')

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net.setInput(blob)
output1 = net.forward()

output1 = output1.squeeze().transpose((1, 2, 0))

output1 += MEAN_VALUE
output1 = np.clip(output1, 0, 255)
output1 = output1.astype('uint8')

net = cv2.dnn.readNetFromTorch('Week2/models/instance_norm/la_muse.t7')

blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net.setInput(blob)
output2 = net.forward()

output2 = output2.squeeze().transpose((1, 2, 0))

output2 += MEAN_VALUE
output2 = np.clip(output2, 0, 255)
output2 = output2.astype('uint8')

output1 = output1[:,:250]
output2 = output2[:,250:]

output = np.concatenate([output1,output2], axis=1)

cv2.imshow('output1', output1)
cv2.imshow('output2', output2)
cv2.imshow('output', output)
cv2.waitKey(0)