import numpy as np
import cv2 as cv
import mediapipe as mp
import autopy
from helper import *

LEFT_EYE_INDICIES = [362, 382, 381, 380, 374, 373,
                     390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICIES = [33, 7, 163, 144, 145, 153,
                      154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS_INDICIES = [474, 475, 476, 477]
RIGHT_IRIS_INDICIES = [469, 470, 471, 472]

SWIDTH, SLENGTH = autopy.screen.size()

scale = 10
smoothening = 2.5
plocX,plocY = 0,0
clocX,clocY = 0,0


mp_face_mesh = mp.solutions.face_mesh
cam = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)  # convert to mirror img
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)  # bgr --> rgb
        height, width = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [width, height]).astype(int)
                                    for p in results.multi_face_landmarks[0].landmark])
            cv.polylines(frame, [mesh_points[LEFT_EYE_INDICIES]], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE_INDICIES]], True, (0, 255, 0), 1, cv.LINE_AA)
            (left_x, left_y), left_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS_INDICIES])
            (right_x, right_y), right_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS_INDICIES])
            left_center = np.array([left_x, left_y], dtype=np.int32)
            right_center = np.array([right_x, right_y], dtype=np.int32)
            avg_x, avg_y = avg(left_x, right_x), avg(left_y, right_y)
            cv.circle(frame, left_center, int(left_radius), (0, 255, 0), 1, cv.LINE_AA)
            cv.circle(frame, right_center, int(right_radius), (0, 255, 0), 1, cv.LINE_AA)

            ratio = blinkRatio(mesh_points,RIGHT_EYE_INDICIES,LEFT_EYE_INDICIES)
            if ratio[0]>5 and ratio[1]<5: #left eye closes
                autopy.mouse.click()
            if ratio[1]>6 and ratio[0]<6: #right eye closes
                autopy.mouse.click(autopy.mouse.RIGHT)

            (leye_x,leye_y), leye_r = cv.minEnclosingCircle(mesh_points[LEFT_EYE_INDICIES])
            (reye_x, reye_y), reye_r = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_INDICIES])
            lu_eyelid = np.multiply(results.multi_face_landmarks[0].landmark[223].y, height)
            lb_eyelid= np.multiply(results.multi_face_landmarks[0].landmark[230].y, height)
            ru_eyelid = np.multiply(results.multi_face_landmarks[0].landmark[443].y, height)
            rb_eyelid = np.multiply(results.multi_face_landmarks[0].landmark[450].y, height)
            cv.circle(frame, np.array([leye_x,leye_y],dtype=np.int32), int(leye_r), (255,0,0), 1, cv.LINE_AA)
            cv.circle(frame, mesh_points[257], 5, (255, 0, 0), cv.FILLED)
            screen_x = avg(np.interp(left_x, (leye_x-leye_r+1.8*left_radius,leye_x+leye_r-1.8*left_radius), (0, SWIDTH)),np.interp(right_x, (reye_x-reye_r+1.8*right_radius,reye_x+reye_r-1.8*right_radius), (0, SWIDTH)))
            screen_y = avg(np.interp(left_y, (lu_eyelid+leye_r-left_radius,lb_eyelid-leye_r+left_radius), (0,SLENGTH)),np.interp(right_y, (ru_eyelid+reye_r-right_radius,rb_eyelid-reye_r+right_radius), (0,SLENGTH)))
            clocX = plocX + (screen_x-plocX)/smoothening
            clocY = plocY + (screen_y-plocY)/smoothening
            autopy.mouse.move(clocX, clocY)
            plocX,plocY = clocX, clocY

        cv.imshow('img', frame)
        if cv.waitKey(1) == ord('q'):
            break
cam.release()
cv.destroyAllWindows()
