import threading
from queue import Queue
import cv2
import numpy as np

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

subframe_size = 200


class ImageProcessor:
    num_frames = 0
    num_optim_frames = 0
    next_point = None
    prev_point = None
    history_points = []

    def processSubFrame(self, w_img):
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

    def drawHistoryPoints(self, img, history_points):
        pp = None
        for p in self.history_points:
            if pp is not None and p is not None:
                cv2.line(img, pp, p, (0, 255, 0), 2)
            pp = p

    def processFrame(self, img):
        self.num_frames += 1

        if self.next_point is not None:
            origin = (max(0, self.next_point[0] - (subframe_size >> 1)), max(0, self.next_point[1] - (subframe_size >> 1)))
            w_img = img[origin[1]:min(len(img), origin[1] + subframe_size),
                    origin[0]:min(len(img[0]), origin[0] + subframe_size)]
            point = self.processSubFrame(w_img)
            # cv2.imshow('Subframe', w_img)
            if point is None:
                point = self.processSubFrame(img)
            else:
                self.num_optim_frames += 1
                point = (point[0] + origin[0], point[1] + origin[1])
        else:
            point = self.processSubFrame(img)

        if point is None:
            self.prev_point = None
            self.next_point = None
        else:
            if self.prev_point is not None:
                self.next_point = (point[0] * 2 - self.prev_point[0], point[1] * 2 - self.prev_point[1])
                if self.next_point[0] < 0 or self.next_point[0] > len(img[0]) or self.next_point[1] < 0 or self.next_point[1] > len(img):
                    self.next_point = point
            self.prev_point = point

        self.history_points.append(point)

        # self.drawHistoryPoints(img, history_points)
        #
        # cv2.imshow('Frame', img)
        #
        # cv2.waitKey(1)


class ImageGrabberThread(threading.Thread):
    def __init__(self, cap, queue):
        threading.Thread.__init__(self)
        self.cap = cap
        self.queue = queue

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read();
            if ret != True:
                break
            self.queue.put(frame)


class ImageProcessorThread(threading.Thread):
    def __init__(self, processor, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.processor = processor
        self.running = True

    def run(self):
        while self.running:
            if not self.queue.empty():
                self.processor.processFrame(self.queue.get())

    def close(self):
        self.running = False


processor = ImageProcessor()

single_thread = True

if single_thread:
    while cap.isOpened():
        ret, img = cap.read();
        if ret == True:
            processor.processFrame(img)
        else:
            break
else:
    frames = Queue(10)
    grabber = ImageGrabberThread(cap, frames)
    main = ImageProcessorThread(processor, frames)
    grabber.start();
    main.start();
    grabber.join();
    main.close();

print("Processed ", + processor.num_frames, " of which ", processor.num_optim_frames, " were adequately predicted")
