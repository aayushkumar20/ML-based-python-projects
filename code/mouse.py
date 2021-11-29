import cv2
import numpy as np
from sklearn.metrics import pairwise

cap=cv2.Videocapture(0)

kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
lb=np.array([20,100,100])
ub=np.array([120,255,255])
while true:
    ret,frame=cap.read()
    flipped=cv2.flip(frame,1)
    flipped=cv2.resize(flipped,(500,500))

    imgSeg=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    imgSegFlipped=cv2.flip(imgSeg,1)
    imgSegFlipped=cv2.resize(imgSegFlipped,(500,400))


    mask=cv2.inRange(imgSegFlipped,lb,ub)
    mask=cv2.resize(mask,(500,400))

    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskOpen=cv2.resize(maskOpen,(500,400))
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    maskClose=cv2.resize(maskClose,(500,400))

    final=maskClose
    _,conts,h=cv2.findContours(maskClose,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if(len(conts)!=0):
        b=max(conts,key=cv2.contourArea)
        west=tuple(b[b[:,:,0].argmin()][0])
        east=tuple(b[b[:,:,0].argmax()][0])
        north=tuple(b[b[:,:,0].argmin()][0])
        south=tuple(b[b[:,:,0].argmax()][0])
        center_x=(west[0]+east[0])/2
        center_y=(north[0]+south[0])/2

        cv2.drawContours(flipped,b,-1,(0,255,0),3)
        cv2.circle(flipped,west,7,(0,0,255),-1)
        cv2.circle(flipped,east,7,(0,0,255),-1)
        cv2.circle(flipped,north,7,(0,0,255),-1)
        cv2.circle(flipped,south,7,(0,0,255),-1)
        cv2.circle(flipped,(int(center_x),int(center_y)),7,(0,0,255),-1)
    
    cv2.imshow('video',flipped)
    #cv2.imshow('mask',mask)
    #cv2.imshow('maskOpen',maskOpen)
    #cv2.imshow('maskClose',maskClose)

    if cv2.waitKey(1)&0xFF==ord(''):
        break

cap.release()
cv2.destroyAllWindows()