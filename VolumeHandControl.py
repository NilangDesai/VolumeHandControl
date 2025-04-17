import cv2
import time
import numpy as np
from jax import devices
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities,IAudioEndpointVolume



import HandTrackingModule as htm
import math
wcam,hcam=640,480

cap=cv2.VideoCapture(1)
cap.set(3,wcam)
cap.set(4,hcam)
pTime=0

detector=htm.HandDetector(detectionCon=0.7)

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
#volume.GetMute()
#volume.GetMasterVolumeLevel()
volumerang= volume.GetVolumeRange()
#print(volumerang)

minvol=volumerang[0]
maxvol=volumerang[1]
vol=0
volbar=400
volper=0

while True:
    success,img=cap.read()
    img=detector.findhands(img)
    lmlist,_=detector.findposition(img,draw=False)
    if len(lmlist) !=0:
        print(lmlist[4],lmlist[8])
        x1,y1=lmlist[4][1],lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img,(x1,y1),15, (255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),2)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        length=math.hypot(x2-x1,y2-y1)
        print(length)

        vol=np.interp(length,[15,240],(minvol,maxvol))
        volbar=np.interp(length,[15,240],(400,150))
        volper=np.interp(length,[15,240],(0,100))
        #print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length<50:
            cv2.circle(img,(cx,cy),15,(0,255,0),2,cv2.FILLED)


    cv2.rectangle(img,(50,150),(85,400),(0,255,0),2)
    cv2.rectangle(img, (50,int(volbar)),(85, 400),(0, 255, 0),cv2.FILLED)
    cv2.putText(img,f'{int(volper)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)