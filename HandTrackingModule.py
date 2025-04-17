import cv2
import mediapipe as mp
import time
import math
import numpy as np




class HandDetector():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackingCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackingCon=trackingCon

        self.mHands = mp.solutions.hands
        self.hands = self.mHands.Hands(static_image_mode=self.mode,
                                       max_num_hands=self.maxHands,
                                       min_detection_confidence=self.detectionCon,
                                       min_tracking_confidence=self.trackingCon)

        self.mpDraw = mp.solutions.drawing_utils
        self.tipids=[4,8,12,16,20]


    def findhands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mHands.HAND_CONNECTIONS)
        return img

    def findposition(self,img,handNo=0,draw=True,rec=True):
        xList=[]
        yList=[]
        bbox=[]
        self.lmList=[]
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myhand.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        if xList and yList:
            xmin,xmax=min(xList),max(xList)
            ymin,ymax=min(yList),max(yList)
            bbox=xmin,xmax,ymin,ymax

            if rec:
                cv2.rectangle(img,(xmin-20,ymin-20),(xmax+20,ymax+20),(0,255,0),2)

        return self.lmList,bbox

    def fingersup(self):
        fingers=[]
        if self.lmList[self.tipids[0]][1] > self.lmList[self.tipids[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmList[self.tipids[id]][2] < self.lmList[self.tipids[id]-1][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def finddistance(self,img,p1,p2,draw=True,r=15,t=3):
        if len(self.lmList) != 0:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if draw:
                cv2.circle(img,(x1, y1),r, (255, 0, 255),cv2.FILLED)
                cv2.circle(img,(x2, y2),r, (255, 0, 255), cv2.FILLED)
                cv2.line(img,(x1, y1), (x2, y2), (255, 0, 255),t)
                cv2.circle(img, (cx, cy),r, (255, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

            return length,[x1,y1,x2,y2,cx,cy]




def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(1)
    detector=HandDetector()

    while True:
        success, img = cap.read()
        img=detector.findhands(img)
        lmList= detector.findposition(img,draw=False)
        #if len(lmList) != 0:
            #print(lmList)
            #fingers=detector.fingersup()
            #print(fingers)
        length=detector.finddistance(img,4,8)
        #print(length)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Img", img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
