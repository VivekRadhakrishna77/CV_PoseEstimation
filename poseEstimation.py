import cv2 as cv
import time
import poseModel as pm


pTime = 0
cap = cv.VideoCapture('pose_estimation/Videos/vid4.mp4')
detector = pm.PoseDetector()
while True:
    ret, img = cap.read()
    img = detector.findPose(img)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (50, 70), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
    cv.imshow("Image", img)
    cv.waitKey(1)