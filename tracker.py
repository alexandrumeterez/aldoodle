from collections import deque
from imutils.video import VideoStream
import numpy as np
import imutils
import time 
import cv2
import matplotlib.pyplot as plt
import math
from keras.models import load_model

# vs = VideoStream(src=0, resolution=(800, 600)).start()
vs = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype = np.uint8)
show_canvas = False
calculated_histogram = False
points = []
model = load_model("model.h5")
classes = ["bed", "bicycle", "brain", "broccoli", "bus", "cat", "dog", "house", "jacket", "star"]
has_drawn = False

far_points = deque(maxlen=10)
def draw_rect(frame):
    rows, cols, _ = frame.shape
    global hand_rect_one_x, hand_rect_one_y
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 
        9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 
        12 * rows / 20, 12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 
        9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 
        9 * cols / 20, 10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(9):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 255), 2)

    return frame

def calculate_histogram(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    for i in range(9):
        roi[i*10:(i+1)*10, 0:10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i]+10]
    histogram = cv2.calcHist([roi], [0,1], None, [180, 256], [0,180,0,256])
    
    return cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)

def isolate_hand(frame, histogram):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], histogram, [0, 180, 0, 256], 1)
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))

    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    thresh = cv2.merge((thresh, thresh, thresh))
    
    return cv2.bitwise_and(frame, thresh)

def extract_hull(hand_frame):
    gray_img = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 0, 255, 0)
    contour_list, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    index = 0
    for i, cnt in enumerate(contour_list):
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            index = i
    hull = cv2.convexHull(contour_list[index], returnPoints=False)

    return hull, contour_list[index]

def extract_fingertip(frame, histogram):
    hand_frame = isolate_hand(frame, histogram)
    hull, cnt = extract_hull(hand_frame)
    defects = cv2.convexityDefects(cnt, hull)
    moments = cv2.moments(cnt)
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    farthest_x, farthest_y = cx, cy
    farthest_dist = 0
    curr_dist = 0


    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        far = tuple(cnt[f][0])
        
        if far[1] < cy:
            curr_dist = math.sqrt((cx - far[0])**2 + (cy - far[1])**2)
            if curr_dist > farthest_dist:
                farthest_dist = curr_dist
                farthest_x, farthest_y = far[0], far[1]
    farthest_point = (farthest_x, farthest_y)

    far_points.appendleft(farthest_point)
    fx, fy = 0, 0
    for p in far_points:
        fx += p[0]
        fy += p[1]
    farthest_point = (int(fx / len(far_points)), int(fy / len(far_points)))
    cv2.circle(frame,farthest_point, 5, [0,0,255],-1)


    return farthest_point
def extract_drawn_contour(canvas):
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(canvas, 9)
    gaussian = cv2.GaussianBlur(median, (5, 5), 0)
    _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contour_list, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contour_list

while True:
    key = cv2.waitKey(1)
    _, frame = vs.read()
    frame = cv2.flip(frame, 1)
    # frame = cv2.convertScaleAbs(frame, alpha = 1.5)
    
    if frame is None:
        break
    if not calculated_histogram:
        frame = draw_rect(frame)
        if key == ord('z'):
            histogram = calculate_histogram(frame)
            calculated_histogram = True
    else:
        
        point = extract_fingertip(frame, histogram)
        if key == ord('x'):
            has_drawn = False
            points.append(point)
        if len(points) >= 2:
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], color=(0, 255, 0), thickness=5)
                cv2.line(canvas, points[i], points[i+1], color=(255, 255, 255), thickness=5)

        if key == ord(' '):
            contour_list = extract_drawn_contour(canvas)
            max_area = -1
            index = 0
            for i, cnt in enumerate(contour_list):
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    index = i
            cnt = contour_list[index]
            x, y, w, h = cv2.boundingRect(cnt)
            img = canvas[y:y+h, x:x+w]
            img = cv2.resize(img, (28, 28))
            predicted_class = classes[np.argmax(model.predict(np.reshape(img, (1,28,28,1))))]
            
            has_drawn = True
            points = []
            canvas = np.zeros((480, 640, 3), dtype = np.uint8)

    if has_drawn:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, predicted_class, (10, 400), font, 4, (0,255,0), thickness=5, lineType=cv2.LINE_AA)


    cv2.imshow("Frame", frame)
    if show_canvas:
        cv2.imshow("Canvas", canvas)
    if key == ord("q"):
        break