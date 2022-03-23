import cv2
import numpy as np

net1 = cv2.dnn.readNetFromTorch('Week2/models/instance_norm/la_muse.t7')
img = cv2.imread('Week2/imgs/04.jpg')

h, w, c = img.shape

img = cv2.resize(img, dsize=(500, int(h / w * 500)))

MEAN_VALUE = [103.939, 116.779, 123.680]
blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net1.setInput(blob)
output1 = net1.forward()

output1 = output1.squeeze().transpose((1, 2, 0))

output1 += MEAN_VALUE
output1 = np.clip(output1, 0, 255)
output1 = output1.astype('uint8')

net2 = cv2.dnn.readNetFromTorch('Week2/models/instance_norm/starry_night.t7')

blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net2.setInput(blob)
output2 = net2.forward()

output2 = output2.squeeze().transpose((1, 2, 0))

output2 += MEAN_VALUE
output2 = np.clip(output2, 0, 255)
output2 = output2.astype('uint8')

net3 = cv2.dnn.readNetFromTorch('Week2/models/instance_norm/candy.t7')

blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net3.setInput(blob)
output3 = net3.forward()

output3 = output3.squeeze().transpose((1, 2, 0))

output3 += MEAN_VALUE
output3 = np.clip(output3, 0, 255)
output3 = output3.astype('uint8')

net4 = cv2.dnn.readNetFromTorch('Week2/models/instance_norm/udnie.t7')

blob = cv2.dnn.blobFromImage(img, mean=MEAN_VALUE)

net4.setInput(blob)
output4 = net4.forward()

output4 = output4.squeeze().transpose((1, 2, 0))

output4 += MEAN_VALUE
output4 = np.clip(output4, 0, 255)
output4 = output4.astype('uint8')

h , w ,c = img.shape

h = int (h/2)
w = int (w/2)

output1 = output1[:h,:w]
output2 = output2[:h,w:]
output3 = output3[h:,:w]
output4 = output4[h:,w:]

res1 = np.concatenate([output1,output2], axis=1)
res2 = np.concatenate([output3,output4], axis=1)

res = np.concatenate([res1,res2],axis=0)

cv2.imshow('output1', output1)
cv2.imshow('output2', output2)
cv2.imshow('output3', output3)
cv2.imshow('output4', output4)

cv2.imshow('res1', res1)
cv2.imshow('res2', res2)


cv2.imshow('res', res)
cv2.waitKey(0)