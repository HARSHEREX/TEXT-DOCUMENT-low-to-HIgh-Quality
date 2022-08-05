import cv2
import os
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('mlm_280x280.h5')

def predict(img,f_shape,y_block=2): #final_height will be f_shape*y_block
  h,w = img.shape[:2]
  img = cv2.normalize(img,np.zeros((h,w)),0,255,cv2.NORM_MINMAX)
  nh = (f_shape*y_block)
  nw = int((nh/h)*w)
  img = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
  pad_x = (f_shape-nw%f_shape)
  pad = np.zeros((nh,pad_x)).astype(np.uint8)
  final_img = np.hstack([img,pad])
  fh,fw = final_img.shape
  print(fh/f_shape,fw/f_shape,'Total Block/Grids : ',(fh/f_shape)*(fw/f_shape))
  grids = []
  for y in range(fh//f_shape):
    for x in range(fw//f_shape):
        g = final_img[int(y*f_shape):int((y+1)*f_shape),int(x*f_shape):int((x+1)*f_shape)]
        grids.append(np.array([g/255]).reshape(f_shape,f_shape,1))
  grids = np.array(grids)
  ops = model.predict(grids)
  opimg = np.zeros((fh,fw)).astype(np.uint8)
  c=0
  for y in range(fh//f_shape):
    for x in range(fw//f_shape):
        opimg[int(y*f_shape):int((y+1)*f_shape),int(x*f_shape):int((x+1)*f_shape)] = (ops[c,:,:,0]*255).astype(np.uint8)
        c+=1
  opimg = opimg[:,:-pad_x]
  return opimg
