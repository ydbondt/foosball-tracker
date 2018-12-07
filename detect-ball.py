import cv2
import numpy as np
import logging

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("OpenCL enabled: ", cv2.ocl.useOpenCL())
else:
    print("OpenCL not available")

cap = cv2.VideoCapture('IMG_0607-non-HEVC.mov')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

if (cap.isOpened() == False):
    print("Error")

ret, first_frame = cap.read()
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (first_frame.shape[1], first_frame.shape[0]))

lower_orange = np.array([0, 162, 186])
upper_orange = np.array([46, 242, 255])

prev_point = None

while (cap.isOpened() == True):
    ret, img = cap.read()
    if ret == True:
        frame = cv2.UMat(img)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        rangeMask = cv2.inRange(hsv, lower_orange, upper_orange)

        image, contours, hierarchy = cv2.findContours(rangeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

        for c in contours:
            M = cv2.moments(c)
            perimeter = cv2.arcLength(c, True)
            if M["m00"] != 0 and perimeter > 100:
                cv2.drawContours(img, [c], -1, (0,255,0), 3)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                point = (cX, cY)
                if (prev_point is not None):
                    cv2.line(img, prev_point, point, (0,255,0), 2)
                    prev_point = point
                break

        # out.write(img)
        cv2.imshow('Frame', img);

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
