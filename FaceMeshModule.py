#To detect 468 landmarks on face

import cv2
import mediapipe as mp
import time

#3.Create a class and define methods
class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, refineLm=False,
                 minDetCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.refineLm = refineLm
        self.minDetCon = minDetCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,
                                                 self.refineLm,self.minDetCon,
                                                 self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self,img, draw= True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          self.drawSpec, self.drawSpec)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    iw, ih, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #print(id,x, y)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #             1, (0, 255, 0), 1)
                    face.append([x,y])
                faces.append(face)
        return img,faces

#2. Define main function
def main():
    cap = cv2.VideoCapture("Videos/1.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img,faces = detector.findFaceMesh(img)
        if len(faces)!=0 :
            print(faces[0])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

#1. if this file is run , then code inside main will run
if __name__ == "__main__":
    main()
