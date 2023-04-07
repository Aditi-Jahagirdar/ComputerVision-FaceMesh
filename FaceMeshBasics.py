#To detect 468 landmarks on face

import cv2
import mediapipe as mp
import time

#1. To run video
cap = cv2.VideoCapture("Videos/2.mp4")
pTime = 0

#3. To find and draw points on faces using mediapipe library
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2)

while True:
    success, img = cap.read()
#4. Convert BGR(cv2) to RGB (mediapipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
#5.To find out and print different points/landmarks in pixel form
            for id,lm in enumerate(faceLms.landmark):
                #print(lm)
                iw, ih, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id,x, y)

#2. To calculate and display frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 3)

    cv2.imshow("Image",img)
    cv2.waitKey(1)


