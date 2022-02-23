import cv2 as cv
import mediapipe as mp
import time



class PoseDetector:
    def __init__(self, mode=False, complexity=1, smooth=True, minDetection=0.5, minTracking=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.minDetection = minDetection
        self.minTracking = minTracking

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, self.minDetection, self.minTracking)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw = True):
        
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw: 
                for id, lm in enumerate(self.results.pose_landmarks.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    cv.circle(img, (cx,cy), 5, (255,0,255), -1)
                
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img
    

def main():
    pTime = 0
    cap = cv.VideoCapture('/Users/vivekradhakrishna/Documents/fcc/Python/CV/opencv-projects/pose_estimation/Videos/vid3.mp4')
    detector = PoseDetector()
    while True:
        ret, img = cap.read()
        img = detector.findPose(img)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (50, 70), cv.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
        cv.imshow("Image", img)
        cv.waitKey(1)

if __name__=='__main__':
    main()