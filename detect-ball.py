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
next_point = None

subframe_size = 200

history_points = []

def processFrame(w_img, prev_point):
    frame = cv2.UMat(w_img)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rangeMask = cv2.inRange(hsv, lower_orange, upper_orange)
    image, contours, hierarchy = cv2.findContours(rangeMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
    for c in contours:
        M = cv2.moments(c)
        perimeter = cv2.arcLength(c, True)
        if M["m00"] != 0 and perimeter > 100:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            point = (cX, cY)
            return point
    return None


def drawHistoryPoints(img, history_points):
    pp = None
    for p in history_points:
        if pp is not None and p is not None:
            cv2.line(img, pp, p, (0, 255, 0), 2)
        pp = p

num_frames = 0
num_optim_frames = 0

while (cap.isOpened() == True):
    ret, img = cap.read()
    if ret == True:
        origin = None

        num_frames += 1

        if next_point is not None:
            origin = (max(0, next_point[0] - (subframe_size >> 1)), max(0, next_point[1] - (subframe_size >> 1)))
            w_img = img[origin[1]:min(len(img), origin[1] + subframe_size), origin[0]:min(len(img[0]), origin[0] + subframe_size)]
            point = processFrame(w_img, prev_point)
            # cv2.imshow('Subframe', w_img)
            if point is None:
                point = processFrame(img, prev_point)
            else:
                num_optim_frames += 1
                point = (point[0] + origin[0], point[1] + origin[1])
        else:
            origin = (0,0)
            point = processFrame(img, prev_point)

        if point is None:
            prev_point = None
            next_point = None
        else:
            if prev_point is not None:
                next_point = (point[0] * 2 - prev_point[0], point[1] * 2 - prev_point[1])
                if next_point[0] < 0 or next_point[0] > len(img[0]) or next_point[1] < 0 or next_point[1] > len(img):
                    next_point = point
            prev_point = point

        history_points.append(point)

        drawHistoryPoints(img, history_points)

        # out.write(img)
        cv2.imshow('Frame', img);

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()

print("Processed ", + num_frames, " of which ", num_optim_frames, " were adequately predicted")