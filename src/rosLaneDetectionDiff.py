#!/usr/bin/env python
import cv2
import numpy as np

import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    #print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
import math
from matplotlib.figure import Figure
from ransac_1d import *
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from matplotlib.backends.backend_agg import FigureCanvasAgg
# plt.get_backend()

IMAGE_W = 640
IMAGE_H = 480
BEV_H = 220
BEV_W = 640

model_Lx = []
model_Ly = []
model_Rx = []
model_Ry = []
Rx = []
Ry = []
Lx = []
Ly = []


centerPubS = 0
centerArray = []

left_bias, right_bias = 330,350

bridge = CvBridge()
tempImage = cv2.imread("/home/jetsontx2/Self-Driving-Delivery-Robot/src/hg_lineDetection/src/testImage/black.jpeg")
rosTempImage = bridge.cv2_to_imgmsg(tempImage)

# # rosRawImg = rosTempImage
rosBevImg = rosTempImage
# # rosFilteredImg = rosTempImage
# # roslROIImg = rosTempImag

# # rosrROIImg= rosTempImage
rosPltImg = rosTempImage

forward = 20

count = 0
centerPub = 0
ldiff = 0
rdiff = 0

pre_l_model_a, pre_l_model_c, pre_r_model_a, pre_r_model_c = 0,0,0,0

img_msg_name = "/stereo/left/image_raw"

def getImage(msg):

    #print("Callback, Get Image")

    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg)

    # img = cv2.imread('/home/jetsontx2/Self-Driving-Delivery-Robot/src/hg_lineDetection/src/testImage/raw/raw_190_img.bmp')

    # img = cv2.resize(img, dsize=(640, 480))
    return img

def getBEV(img):

    #print("TransForm to BEV image")

    #Select ROI Range
    p1 = [0, 200]
    p2 = [640, 200]
    p3 = [640, 480]
    p4 = [0, 480]

    corner_points_arr = np.float32([p1, p2, p3, p4])
    height, width = img.shape[:2]

    image_p1 = [0, 0]
    image_p2 = [width, 0]
    image_p3 = [width, height]
    image_p4 = [0, height]

    image_params = np.float32([image_p1, image_p2, image_p3, image_p4])

    mat = cv2.getPerspectiveTransform(corner_points_arr, image_params)

    global revM

    # Inverse transformation
    revM = cv2.getPerspectiveTransform(image_params, corner_points_arr)  

    image_transformed = cv2.warpPerspective(img, mat, (width, height))
    
    return image_transformed

def filterLine(bevImg):

    #print("Filtering BEV image")

    #get hsv from bev image
    hsv = cv2.cvtColor(bevImg, cv2.COLOR_BGR2HSV)    # re_img = image

    # Filter white pixels
    white_threshold = 135  # 130
    lower_white = np.array([white_threshold, white_threshold, white_threshold])
    upper_white = np.array([180, 255, 255])

    # create white mask using threshold array
    white_mask = cv2.inRange(bevImg, lower_white, upper_white)

    # remove noise, opening
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), dtype=np.uint8))

    # close mask
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), dtype=np.uint8))

    # using bilateralFilter, bluring without edge area
    white_mask = cv2.bilateralFilter(white_mask, 5, 100, 100);

    # bitwise white image to bevImage
    white_image = cv2.bitwise_and(bevImg, bevImg, mask=white_mask)
    
    # Filter yellow pixels
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # create yellow mask using threshold array
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # # remove noise, opening
    # yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel=np.ones((3, 3), dtype=np.uint8))

    # # close mask
    # yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), dtype=np.uint8))

    # # using bilateralFilter, bluring without edge area
    # yellow_mask = cv2.bilateralFilter(yellow_mask, 5, 100, 100);

    # bitwise yellow image to bevImage
    yellow_image = cv2.bitwise_and(bevImg, bevImg, mask=yellow_mask)

    # Filter blue pixels
    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])

    # create blue mask using threshold array
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # bitwise blue image to bevImage
    blue_image = cv2.bitwise_and(bevImg, bevImg, mask=blue_mask)

    # Filter red pixels
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([30, 255, 255])

    # create red mask using threshold array
    red_mask = cv2.inRange(hsv, lower_red, upper_red)

    # bitwise blue image to bevImage
    red_image = cv2.bitwise_and(bevImg, bevImg, mask=red_mask)

    # add white, blue, yellow, red images weight to image
    null_img = np.zeros((bevImg.shape[0], bevImg.shape[1], 3), dtype=np.uint8)
    re_img = cv2.addWeighted(white_image, 1., null_img, 1., 0.)
    re_img = cv2.addWeighted(yellow_image, 1., re_img, 1., 0.)
    re_img = cv2.addWeighted(blue_image, 1., re_img, 1., 0.)
    re_img = cv2.addWeighted(red_image, 1., re_img, 1., 0.)

    return re_img

def regionOfInterest(img, vertices, color3=(255, 255, 255), color1=255):

    #print("Set ROI range")

    mask = np.zeros_like(img)

    # fill not ROI area with black poly
    cv2.fillPoly(mask, vertices, color3)

    # bitwise mask to image
    ROI_image = cv2.bitwise_and(img, mask)

    return ROI_image

def gethalfROIimage(bevImg,lr):

    #print("Get Half {lr} ROI image")

    # to use remove ROI area edges
    addPixel = 20

    def __init__():
        vertices

    if lr == "left":
        vertices = np.array(
                        [
                            [
                                (0, 0),
                                (240 + addPixel, 0),
                                (240 + addPixel, IMAGE_H / 2 + addPixel),
                                (0, IMAGE_H / 2 + addPixel)
                            ]
                        ],
                        dtype=np.int32)
    # set ROI vertices
    elif lr == "right":
        vertices = np.array(
                        [
                            [
                                (400 - addPixel, 0),
                                (IMAGE_W, 0),
                                (IMAGE_W, IMAGE_H / 2 + addPixel),
                                (400 - addPixel, IMAGE_H / 2 + addPixel)
                            ]
                        ],
                        dtype=np.int32)
    else:
        #print("parameter Exception")
        pass

    # get ROI vertices
    halfROIImg = filterLine(regionOfInterest(bevImg, vertices))

    # filtering to half_ROI_img with Gaussianblur
    gray_img = cv2.cvtColor(halfROIImg, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0) # gaussian Blur

    # get edges using canny func
    halfROIImg = cv2.Canny(blur_img, 100, 150)  # Canny edge

    # set ROI vertices removing ROI area edges
    if lr == "left":
        vertices = np.array(
            [
                [
                    (0, 0),
                    (240, 0),
                    (240, IMAGE_H / 2),
                    (0, IMAGE_H / 2)
                ]
            ],
            dtype=np.int32)
    elif lr == "right":
        vertices = np.array(
            [
                [
                    (400, 0),
                    (IMAGE_W, 0),
                    (IMAGE_W, IMAGE_H / 2),
                    (400, IMAGE_H / 2)
                ]
            ],
            dtype=np.int32)
    else:
        #print("parameter Exception")
        pass

    halfROIimg = regionOfInterest(halfROIImg, vertices)

    return halfROIimg

def getHoughLines(roiImg):

    #print("Get HoughLines")

    # get Houghlines
    lines = cv2.HoughLinesP(roiImg, 1, 1 * np.pi / 180, 30, minLineLength=30, maxLineGap=10)
    lines = np.squeeze(lines)

    return lines

# def getHough



def drawLines(img, lines, color=[0, 0, 255], thickness=7):
    #print("draw line to image")

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def getLineDrawedImage(bevImg, lLines, rLines):

    #print("get edge image")

    # create line palette
    line_img = np.zeros((bevImg.shape[0], bevImg.shape[1], 3), dtype=np.uint8)

    # check line dimention size and draw lines
    if lLines.ndim != 0:
        drawLines(line_img, np.array(lLines)[:, None])
    if rLines.ndim != 0:
        drawLines(line_img, np.array(rLines)[:, None])

    # add lines to bevImg
    edge_img = cv2.addWeighted(bevImg, 1, line_img, 1, 0)

    return edge_img

def interpolate(x1, y1, x2, y2):
    re = []
    if abs(y2 - y1) < 2:
        return re

    re.append([x1, y1])
    re.append([x2, y2])

    if (x2 - x1) == 0:
        tilt = 10000
    else:
        tilt = (y2 - y1) / (x2 - x1 + 0.000001)
    for i in range(x1, x2, 1):
        idx = i - x1
        re.append([x1 + idx, y1 + tilt * idx])
    return re

def getModel(line_arr, arrX, arrY):
    for i in line_arr:
        reArray = interpolate(i[0], i[1], i[2], i[3])
        if len(reArray) == 0:
            ttemp = 1
        else:
            for re in reArray:
                arrX.append(re[0])
                arrY.append(re[1])

    model_a, model_b, model_c, max = RANSAC(arrX, arrY)

    # #print("LINE {}, Point {} Inlier {}".format(len(line_arr), len(arrX), max))
    return model_a, model_c, max

def getXBias(model_a, model_c, height):
    model_bias = (height+model_c)/(model_a + 0.000001)
    return model_bias

def canUseModel(model_a, model_c, pre_model_a, pre_model_c, inlierRatio):
    if model_a == 0:
        #print("NO MODEL")
        return False

    model_bias = getXBias(model_a, model_c, 200)
    pre_model_bias = getXBias(pre_model_a, pre_model_c, 200)

    sub = abs(model_bias - pre_model_bias)
    # #print("SUB = {}".format(sub))
    # #print("BIAS = {}".format(model_bias))

    if pre_model_a == 0 or inlierRatio > 0.4:
        return True
    if sub > 100:
        return False

    return True

def lineEquation(x1, y1, x2, y2):

    #print("get center")

    global m, b, x

    # calculate slope
    m = (y2 - y1) / (x2 - x1)

    b = y1 - m * x1

    global center

    # get center dot (center,50)
    center = (10 - b)/m

    # set x range to image x pixel range(0~640)
    x = np.array(range(0,640))

    #print("center line: y = {m}x + {b}")

    return center

def getLineSlope(x1,x2,y1 = 20, y2 = 220):

    #print("get line slope")

    y_diff = y2-y1
    x_diff = x2-x1

    #print(x1,y1,x2,y2)

    return y_diff / x_diff

def printLane(bevImg, la, lc, ra, rc, forward,islLine,isrLine):

    #print("print Lane")

    center = 0
    color=(255,0,0)

    lx = (forward - lc)/(la+0.00001)
    rx = (forward - rc)/(ra+0.00001)

    lx2 = (BEV_H-lc)/(la+0.00001)
    rx2 = (BEV_H-rc)/(ra+0.00001)

    # get line slopes

    lLineSlope = getLineSlope(lx, lx2)
    rLineSlope = getLineSlope(rx, rx2)

    # line slopes test
    # if(islLine!=False):  #print(lLineSlope)
    # if(isrLine!=False):  #print(rLineSlope)

    # use to filtering error line
    slopeValue = 0.2

    # filter error line
    # lLine error
    if (-rLineSlope + slopeValue <= lLineSlope):  # case Left line is bent to the right
        islLine = False
    # if(rLineSlope - slopeValue >= lLineSlope): # case Left line is bent to the right
    #     islLine = False
    if (lLineSlope < slopeValue and lLineSlope > -slopeValue):  # case Left line is drawed horizontally
        islLine = False
    # if(lLineSlope - slopeValue >= rLineSlope): # case Right line is bent to the right
    #     isrLine = False
    if (-lLineSlope - slopeValue >= rLineSlope):  # case Right line is bent to the left
        isrLine = False
    if (rLineSlope < slopeValue and rLineSlope > -slopeValue):  # case Right line is drawed horizontally
        isrLine = False

    # check Line exist
    if (isrLine == False and islLine == False):

        # no lines, create perpendicular line to x = 320
        lx = 159
        rx = 481
        lx2 = 1
        rx2 = 641
        left_diff = 10000
        right_diff = 10000

    elif(islLine==False):
        # lx = rx - 500
        # lx2 = rx2 - 900
        left_diff = 10000
        right_diff = rx - 320
    elif(isrLine==False):
        # rx = lx + 500
        # rx2 = lx2 + 900
        left_diff = lx - 320
        right_diff = 10000
    else:
        left_diff = lx - 320
        right_diff = rx - 320
    # center = lineEquation((lx2 + rx2)/2,220,(lx + rx)/2,20)
 

    # vertice = np.array(
    #     [
    #         [
    #             (int(lx), forward),
    #             (int(rx), forward),
    #             (int(rx2), BEV_H),
    #             (int(lx2), BEV_H)
    #         ]
    #     ]
    # )

    # # create segementation mask
    # nor_mask = np.zeros_like(bevImg)
    # cv2.fillPoly(nor_mask, vertice, color)  #(255,0,0)

    # BEV = cv2.warpPerspective(nor_mask, revM, (BEV_W, BEV_H)) # get BEV

    # bitwise segementation mask to bevImg
    # segImage = cv2.bitwise_or(bevImg, nor_mask)

    # return center, lLineSlope, rLineSlope
    return left_diff, right_diff

def getResult(rawImg, bevImg, leftLine,rightLine):
    
    #print("get result using Ransac")

    # get model

    global model_Lx, model_Ly, model_Rx, model_Ry, Rx, Ry, Lx, Ly, forward, pre_l_model_a, pre_l_model_c, pre_r_model_a, pre_r_model_c, left_bias, right_bias, center

    model_Lx[:], model_Ly[:], model_Rx[:], model_Ry[:], Rx[:], Ry[:], Lx[:], Ly[:] = [], [], [], [], [], [], [], []

    l_model_a, l_model_c, r_model_a, r_model_c = 0, 0, 0, 0

    # to check line exist
    islLine = True
    isrLine = True

    if leftLine.size > 4:
        l_model_a, l_model_c, inlier = getModel(leftLine, Lx, Ly)
        # #print("LEFT -----------------------------")
        if canUseModel(l_model_a, l_model_c, pre_l_model_a, pre_l_model_c, inlier / len(leftLine)) == False:
            l_model_a = pre_l_model_a
            l_model_c = pre_l_model_c

        model_Lx = [a for a in range(0, 600)]
        model_Ly = [l_model_a * a + int(l_model_c) for a in range(0, 600)]

        if len(model_Ly) != len(model_Lx):
            #print("XY {} {}".format(len(model_Lx), len(model_Ly)))
            #print("model {} {}".format(l_model_a, l_model_c))
            while True:
                l_model_a = l_model_a

        pre_l_model_a = l_model_a
        pre_l_model_c = l_model_c
    else:
        islLine = False
    

    if rightLine.size > 4:
        r_model_a, r_model_c, inlier = getModel(rightLine, Rx, Ry)
        # #print("RIGHT -----------------------------")

        if canUseModel(r_model_a, r_model_c, pre_r_model_a, pre_r_model_c, inlier / len(rightLine)) == False:
            r_model_a = pre_r_model_a
            r_model_c = pre_r_model_c

        model_Rx = [a for a in range(0, 600)]
        model_Ry = [r_model_a * a + int(r_model_c) for a in range(0, 600)]

        if len(model_Ry) != len(model_Rx):
            #print("XY {} {}".format(len(model_Rx), len(model_Ry)))
            #print("model {} {}".format(r_model_a, r_model_c))
            while True:
                r_model_a = r_model_a

        pre_r_model_a = r_model_a
        pre_r_model_c = r_model_c
    else:
        isrLine = False

    # center, lslope, rslope = printLane(bevImg, l_model_a, l_model_c, r_model_a, r_model_c, forward,islLine,isrLine)
    left_diff, right_diff = printLane(bevImg, l_model_a, l_model_c, r_model_a, r_model_c, forward,islLine,isrLine)

    # return center, lslope, rslope
    return left_diff, right_diff

def figure_to_array(fig):
    fig.canvas.draw()
    return np.array(fig.canvas.renderer._renderer)

def getPlt(cen):

    #print("get Plt result image")

    global model_Lx, model_Ly, model_Rx, model_Ry, Rx, Ry, Lx, Ly

    plt.cla()
    
    if len(model_Lx) > 4:
        plt.plot(model_Lx, model_Ly, '-r')
        plt.plot(Lx, Ly, '*k')
    if len(model_Rx) > 4:
        plt.plot(model_Rx, model_Ry, '-g')
        plt.plot(Rx, Ry, '*b')

    plt.plot(x,m*x+b)
    plt.text(320, 320, str(cen))
    plt.xlim(0, 640)
    plt.ylim(480, 0)
    plt.grid(True)

    #save graph image
    plt.savefig("/home/jetsontx2/Self-Driving-Delivery-Robot/src/hg_lineDetection/src/testImage/plt/"+str(count)+str(pltCount)+ "_img.png")

    return

def getDirection(center):

    #print("calculate difference to center")

    return center - 320

def save_image(img,path):
    path = "/home/jetsontx2/Self-Driving-Delivery-Robot/src/hg_lineDetection/src/testImage/" + path + str(count) + "_img.bmp"
    #print(path)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))    

def imageCallback(msg):
    rosRawImage_publish.publish(msg)


    global centerPub
    global count
    global pltCount


    global ldiff
    global rdiff 

    pltCount = 0

    rawImg = getImage(msg)
    bevImg = getBEV(rawImg)
    filteredImg = filterLine(bevImg)
    leftROIImg = gethalfROIimage(filteredImg, "left")
    rightROIImg = gethalfROIimage(filteredImg, "right")
    lLines = getHoughLines(leftROIImg)
    rLines = getHoughLines(rightROIImg)
    # edgeImg = getLineDrawedImage(bevImg, leftROIImg, rightROIImg)

    # to calculate center average
    # centerSum = 0

    lsum = 0 
    rsum = 0 

    lslopeSum = 0
    rslopeSum = 0
    lsCount = 0
    rsCount = 0
    
    for x in range(3):
        pltCount+=1
        # center, lslope, rslope = getResult(rawImg, bevImg, lLines, rLines)

        left_diff, right_diff = getResult(rawImg, bevImg, lLines, rLines)

        # centerSum += getDirection(center)
        # centerSum += center
        lsum += left_diff
        rsum += right_diff

        # if lslope is not None:
        #     # print(lslope)
        #     lslopeSum += lslope
        #     lsCount += 1
        # if rslope is not None:
        #     # print(rslope)
        #     rslopeSum += rslope
        #     rsCount+=1
    
    # getPlt()

    count+=1

    if lsCount > 0:
        lslopeSum = lslopeSum/lsCount

    if rsCount > 0:
        rslopeSum = rslopeSum/rsCount

    # centerPub = centerSum/3
    # centerPub = center

    ldiff = lsum/3
    rdiff = rsum/3

    # if slope < abs(1) have to turn to folling direction

    # temp = 200

    
    # global centerPubS

    # centerPubS = 0


    # if lslopeSum < -0.3 and lslopeSum > -0.8:
    #     slopeLCenter = temp * (1 + lslopeSum)
    #     centerPubS = slopeLCenter

    # if rslopeSum > 0.3 and rslopeSum < 0.8:
    #     slopeRCenter = -(temp * (1 - rslopeSum))
    #     centerPubS = slopeRCenter
    
    # #sliding
    # isMinus = False
    # if centerPub < 0:
	# isMinus = True

    # if centerPub > 200: 
	# centerPub=200
    # if centerPub < -200: 
    #     centerPub=-200
    #     isMinus = True

    # if centerPub is not 0:
    #     centerPub = math.log(abs(centerPub), 2)

    # if(isMinus): centerPub = -centerPub

    # centerPub = centerPub * 12
    # # centerArray.append(centerPub)
    # # print(centerPub)



    # slope
    


    #print("************************")
    #print("center average: " + str(centerPub))
    #print("************************")

    # save img data
    # """
    
    # if(count%10==0):
        # save_image(rawImg, "raw/raw_")
        # save_image(bevImg, "bev/bev_")
        # save_image(filteredImg, "filtered/filtered_")
        # save_image(leftROIImg, "ROI/lROI_")
        # save_image(rightROIImg, "ROI/rROI_")
        # getPlt(centerPub)
        # save_image(edgeImg, "edge/edge_")
    # """

    # bridge = CvBridge()

    # pltImg = cv2.imread("/home/jetsontx2/Self-Driving-Delivery-Robot/src/hg_lineDetection/src/testImage/plt/"+str(count)+str(pltCount)+ "_img.png")


    try:
        rosBevImg = bridge.cv2_to_imgmsg(bevImg,encoding="rgb8")
        rosFilteredImg = bridge.cv2_to_imgmsg(filteredImg, encoding="rgb8")
        roslROIImg = bridge.cv2_to_imgmsg(leftROIImg, encoding="mono8")
        rosrROIImg = bridge.cv2_to_imgmsg(rightROIImg, encoding="mono8")
    
        rosBevImage_publish.publish(rosBevImg)
        rosFilteredImage_publish.publish(rosFilteredImg)
        roslROIImage_publish.publish(roslROIImg)
        rosrROIImage_publish.publish(rosrROIImg)
    except Exception:
        pass

def main():
    
    rospy.init_node('laneDiff')
    rate = rospy.Rate(2)

    rospy.Subscriber(img_msg_name,Image, imageCallback, queue_size=1)
    left_publish = rospy.Publisher("lane_left", Float64, queue_size=10)
    right_publish = rospy.Publisher("lane_right", Float64, queue_size=10)

    global rosRawImage_publish, rosRawImage_publish, rosBevImage_publish, rosFilteredImage_publish, roslROIImage_publish, rosrROIImage_publish, rosPltImage_publish

    rosRawImage_publish = rospy.Publisher("/lane/raw_image", Image, queue_size=5)
    rosBevImage_publish = rospy.Publisher("/lane/bev_image", Image, queue_size=5)
    rosFilteredImage_publish = rospy.Publisher("/lane/filter_image", Image, queue_size=5)
    roslROIImage_publish = rospy.Publisher("/lane/left_roi", Image, queue_size=5)
    rosrROIImage_publish = rospy.Publisher("/lane/right_roi", Image, queue_size=5)
    rosPltImage_publish = rospy.Publisher("/lane/plt", Image, queue_size=5)

    # plt.cla()
    while not rospy.is_shutdown():
        # rospy.loginfo("center: " + str(centerPub))
        rospy.loginfo("left_diff: " + str(ldiff))
        rospy.loginfo("right_diff: " + str(rdiff))
        # rospy.loginfo("differ: " + str(centerPubS))
        # rospy.loginfo("slope: " + str(m))
        left_publish.publish(ldiff)
        right_publish.publish(rdiff)
        rate.sleep()

if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass



