# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:27:37 2013

@author: Andrew
"""

import cv2, time, numpy as np, serial as ps

class ChaseBall(object):
    
    def __init__(self):
        
        self.imgScale = np.array([1.0, 1.0 , 1.0], np.float32)        
        
        self.ballFound = False
        self.frontCarFound = False
        self.backCarFound = False        
                
#        self.ballMean = np.array([119.2969, 202.1691, 193.8351])
#        self.carBackMean = np.array([36.5480, 9.8417, 190.5557])
#        self.carFrontMean = np.array([220.8997, 113.6161, 9.2579])
        self.ballMean = np.array([95.2919, 193.6647, 178.0390])
        self.carBackMean = np.array([22.2108, 14.2900, 174.0439])
        self.carFrontMean = np.array([177.9610, 101.8435, 9.8372])
        
#        self.ballCov = np.array([[1562.1, 1732.7, 1370.9],
#                [1732.7, 1961.2, 1513.6], [1370.9, 1513.6, 1228.8]])
#        self.carBackCov = np.array([[90.2170, 28.3257, 233.0151],
#                [28.3257, 13.2580, 68.0371], [233.0151, 68.0371, 668.2910]])
#        self.carFrontCov = np.array([[206.7356, 156.8261, 14.2522],
#                [156.8261, 228.0134, 35.3706], [14.2522, 35.3706, 42.5262]])
        self.ballCov = np.array([[1722.1, 1770.2, 1952.4],
                [1770.2, 2520.8, 2538.9], [1952.4, 2538.9, 2679.1]])
        self.carBackCov = np.array([[213.0, -27.1, 338.8],
                [-27.1, 47.8, 72.2], [338.8, 72.2, 1099.3]])
        self.carFrontCov = np.array([[1974.4, 1026.2, 105.1],
                [1026.2, 750.5, 112.2], [105.1, 112.2, 38.2]])
        
        self.colorThresh = 4
        
        self.vc = cv2.VideoCapture(0)
        
        self.KInv = np.array([[ 0.00201472, 0.0,        -0.64312242],
                              [ 0.0,        0.00203603, -0.3760941 ],
                              [ 0.0,        0.0,         1.0       ]])                     
                              

        h = 3.0 * np.pi / 8.0
        self.H = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(h), -np.sin(h)], 
                            [0.0, np.sin(h), np.cos(h)]])
        
        self.H = np.dot(self.H, self.KInv)
       
        if not self.vc.isOpened():
            
            print 'Video Capture is not working.'
            
            
        cv2.namedWindow('ThresholdedImage', cv2.CV_WINDOW_AUTOSIZE)
#        self.ser = ps.Serial('/dev/cu.usbmodemfa131', 9600)
        self.ser = ps.Serial('/dev/tty.usbmodemfa131', 9600)
#        self.ser = ps.Serial('/dev/cu.usbmodemfd121', 9600)

        
    def onMouse(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1, self.y1 = np.int16([x,y])
            
            
        if event == cv2.EVENT_LBUTTONUP:
            self.x2, self.y2 = np.int16([x,y])
    
    def setImgScale(self):
        
        self.getImg()
        
        bMax = np.amax(self.img[:,:,0])
        gMax = np.amax(self.img[:,:,1])
        rMax = np.amax(self.img[:,:,2])
        
        self.imgScale = np.array([bMax, gMax, rMax], np.float32)
        self.imgScale = 255.0 / self.imgScale
        
    def getImg(self):
        
        rval, img = self.vc.read()
        img = img[0::2, 0::2, :]
        
        self.img = cv2.GaussianBlur(img, (5,5), 1)
        
        self.img = self.img.astype(np.float32)
        self.img[:,:,0] = self.img[:,:,0]  * self.imgScale[0]
        self.img[:,:,1] = self.img[:,:,1]  * self.imgScale[1]
        self.img[:,:,2] = self.img[:,:,2]  * self.imgScale[2]
        
        self.img = np.clip(self.img, 0, 255)        
        self.img = self.img.astype(np.uint8)
        
        self.displayImg = self.img
        
        
    def addColor(self, whichObject):
        
        self.getImg()
        
        if whichObject == 0:
            
            windowStr = 'Pick ball color'
            
        elif whichObject == 1:
            
            windowStr = 'Pick car front color'
            
        else:
            
            windowStr = 'Pick car back color'
        
        cv2.namedWindow(windowStr, cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow(windowStr, self.img)
        cv2.setMouseCallback(windowStr,self.onMouse)
        cv2.waitKey()
        cv2.destroyWindow(windowStr)
        
        if self.x1 < self.x2:
            
            x1 = self.x1
            x2 = self.x2
            
        else:
            
            x1 = self.x2
            x2 = self.x1
            
        if self.y1 < self.y2:
            
            y1 = self.y1
            y2 = self.y2
            
        else:
            
            y1 = self.y2
            y2 = self.y1
            
        
        temp = self.img[y1 : y2, x1 : x2, :]
        numPts = np.shape(temp)[0] * np.shape(temp)[1]
        newTemp = np.reshape(temp, (numPts, 3))
        
#        for idx in range(numPts):
#            
#            print str(newTemp[idx, :]) + ';'
        
        colorMean = np.mean(newTemp, axis = 0)    
        print colorMean
        
        colorCov = np.cov(newTemp.T)
        print colorCov
        
        if whichObject == 0:
            
            self.ballMean = colorMean
            self.ballCov = colorCov
                                    
        elif whichObject == 1:
                                    
            self.carFrontMean = colorMean
            self.carFrontCov = colorCov
                                    
        elif whichObject == 2:
                                    
            self.carBackMean = colorMean
            self.carBackCov = colorCov
        
        
    def findColorObjectMahalanobis(self, color, P):
                
        Pinv = np.linalg.inv(P)
        
        img0 = self.img[:,:,0] - color[0]
        img1 = self.img[:,:,1] - color[1]
        img2 = self.img[:,:,2] - color[2]
        
        temp1 = img0 * Pinv[0,0] + img1 * Pinv[0,1] + img2 * Pinv[0,2]
        temp2 = img0 * Pinv[1,0] + img1 * Pinv[1,1] + img2 * Pinv[1,2]
        temp3 = img0 * Pinv[2,0] + img1 * Pinv[2,1] + img2 * Pinv[2,2]
            
        binImg = 255 - np.sqrt(img0 * temp1 + img1 * temp2 + img2 * temp3)
        binImg = np.clip(binImg, 0, 255).astype(np.uint8)    
            
#        cv2.imshow('ThresholdedImage', binImg)
#        cv2.waitKey()
        
        binImg[binImg > (255 - self.colorThresh)] = 255
        binImg[binImg < 255] = 0
        
#        cv2.imshow('ThresholdedImage', binImg)
#        cv2.waitKey()

        binImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE,
                                  np.ones(5, dtype = np.uint8))
        
        contourImg = binImg.copy()
        contours, hierarchy = cv2.findContours(contourImg, 
                                cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                
        max_area = 0
        best_cnt = np.array([0])

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                best_cnt = cnt
        
        if best_cnt.any():  
                 
            M = cv2.moments(best_cnt)
            cx,cy = M['m10']/M['m00'], M['m01']/M['m00']
                     
        else:
            
            cx, cy = -1, -1
        
        return cx, cy  
                                                

    def findBall(self):
        
        self.ballFound = False
        
        self.getImg()

        cx, cy = self.findColorObjectMahalanobis(self.ballMean, self.ballCov)
                                
        if cx > -1: 
            
            self.ballFound = True
            cv2.circle(self.displayImg, (int(cx), int(cy)), 5, (0, 0, 255), -1)
            
            print 'ball'
            print np.array([cx, cy])
            cx, cy, cz = np.dot(self.H, np.array([cx, cy, 1.0]))
            self.ballLoc = np.array([cx / cz, cy / cz])
            
            
        
    def findCar(self):
        
        self.frontCarFound = False
        self.backCarFound = False

        cx, cy = self.findColorObjectMahalanobis(self.carBackMean, 
                                                 self.carBackCov)
                                
        if cx > -1: 
            
            self.backCarFound = True
            cv2.circle(self.displayImg, (int(cx), int(cy)), 5, (255,0,0), -1)
            
            print 'car back'
            print np.array([cx, cy])
            cx, cy, cz = np.dot(self.H, np.array([cx, cy, 1.0]))
            self.backCarLoc = np.array([cx / cz, cy / cz])
                                
        cx, cy = self.findColorObjectMahalanobis(self.carFrontMean, 
                                                 self.carFrontCov)
    
        if cx > -1: 
            
            self.frontCarFound = True
            cv2.circle(self.displayImg, (int(cx), int(cy)), 5, (255, 0, 255), 
                       -1)

            print 'car front'
            print np.array([cx, cy])
            cx, cy, cz = np.dot(self.H, np.array([cx, cy, 1.0]))
            self.frontCarLoc = np.array([cx / cz, cy / cz])
        
        
    def determineMotion(self):
        
        self.findBall()
        self.findCar()
        
        # forward = 0 - stationary
        # forward = 1 - forward
        # forward = 2 - backward
        # turn = 0 - no turn
        # turn = 1 - left turn
        # turn = 2 - right turn
                
        forward = 0
        turn = 0
        
        if self.ballFound and self.backCarFound and self.frontCarFound:
            
            carDiff = self.frontCarLoc - self.backCarLoc
            carDir = np.arctan2(carDiff[1], carDiff[0])            
            carDist = carDiff[0] * carDiff[0] + carDiff[1] * carDiff[1]        
            
            ballDiff = self.ballLoc - self.frontCarLoc
            ballDir = np.arctan2(ballDiff[1], ballDiff[0])
            ballDist = ballDiff[0] * ballDiff[0] + ballDiff[1] * ballDiff[1] 
            
            
            moveDir = (ballDir - carDir) * 180 / np.pi
            if moveDir > 180:
                
                moveDir = moveDir - 360
                
            if moveDir < -180:
                
                moveDir = moveDir + 360
            
            print moveDir
            print ballDist / carDist
            if np.abs(moveDir) <= 20:
                
                if ballDist > 0.5 * carDist:

                    forward = 1
                    turn = 0
                    
            if moveDir > 20 and moveDir <= 60:
                
                if ballDist > (4 * carDist):
                    
                    forward = 1
                    turn = 1
                    
                else:
                    
                    forward = 2
                    turn = 2
                    
            if moveDir < -20 and moveDir >= -60:
                
                if ballDist > (4 * carDist):
                    
                    forward = 1
                    turn = 2
                    
                else:
                    
                    forward = 2
                    turn = 1
                    
            if moveDir > 60 and moveDir <= 120:
                
                forward = 2
                turn = 2
                
            if moveDir < -60 and moveDir >= -120:
                
                forward = 2
                turn = 1
                
            if moveDir > 120 and moveDir < 180:
                
                if ballDist < (4 * carDist):
                    
                    forward = 1
                    turn = 1
                    
                else:
                    
                    forward = 2
                    turn = 2
                    
            if moveDir < -120 and moveDir >= -180:
                
                if ballDist < (4 * carDist):
                    
                    forward = 1
                    turn = 2
                    
                else:
                    
                    forward = 2
                    turn = 1
                   
        print forward, turn    
        return forward, turn
            
            
                           
    def moveCar(self, forward, turn):
        
        # a = forward
        # b = backward
        # c = forward left
        # d = forward right
        # e = backward right
        # f = backward left
        if forward > 0:
            
            if forward == 1:
                
                if turn == 0:
                    
                    self.ser.write('a')
                    
                elif turn == 1:
                    
                    self.ser.write('d')
                    
                else:
                    
                    self.ser.write('c')
                    
            else:
            
                if turn == 0:
                    
                    self.ser.write('b')
                    
                elif turn == 1:
                    
                    self.ser.write('f')
            
                else:
                    
                    self.ser.write('e')
        
                    

    def displayMoveDirection(self, forward, turn):

        if forward > 0:
            
            if forward < 2:
                
                if turn == 0:
                    
                    ang = 0
                    
                elif turn == 1:
                    
                    ang = np.pi / 4
                    
                else:
                    
                    ang = -np.pi / 4
                    
            else:
                
                if turn == 0:
                    
                    ang = np.pi
                    
                elif turn == 1:
                    
                    ang = 3 * np.pi / 4
                    
                else:
                    
                    ang = -3 * np.pi / 4
            
            carDiff = self.frontCarLoc - self.backCarLoc
            carDir = np.arctan2(carDiff[0], carDiff[1]) 
            ang += carDir            
            
            pt1 = self.frontCarLoc
            Rx = np.cos(ang)
            Ry = np.sin(ang)
            
            ptx = Rx * 0.0 + Ry * 50.0
            pty = Rx * 50.0 - Ry * 0.0
            
            pt2 = self.frontCarLoc + np.array([ptx, pty])
            
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0]), int(pt2[1]))
            cv2.line(self.displayImg, pt1, pt2, (255, 255, 255), 2)


    def estimateHomography(self):
        
        pts1Found = False
        
        
        while not pts1Found:
            
            self.getImg()
            
            fx1, fy1 = self.findColorObjectMahalanobis(self.carFrontMean, 
                                            self.carFrontCov) 
       
            bx1, by1 = self.findColorObjectMahalanobis(self.carBackMean, 
                                            self.carBackCov) 
            
            cv2.imshow('ThresholdedImage', self.displayImg)
            cv2.waitKey()
            pts1Found = (fx1 > -1) and (bx1 > -1)
            
            if not pts1Found:
                
                forward, turn = 1, 0
                self.moveCar(forward, turn)
                time.sleep(0.75)
                self.moveCar(forward, turn)
            
        print 'Found 1st set'    
        pts2Found = False
        
        while not pts2Found:
            
            forward, turn = 1, 0
            self.moveCar(forward, turn)
            time.sleep(0.75)
            self.moveCar(forward, turn)
            
            self.getImg()
            
            fx2, fy2 = self.findColorObjectMahalanobis(self.carFrontMean, 
                                            self.carFrontCov) 
       
            bx2, by2 = self.findColorObjectMahalanobis(self.carBackMean, 
                                            self.carBackCov) 
                                            
            pts2Found = (fx2 > -1) and (bx2 > -1)
        
                                            
        fx1, fy1, fz1 = np.dot(self.KInv, np.array([fx1, fy1, 1.0]))  
        bx1, by1, bz1 = np.dot(self.KInv, np.array([bx1, by1, 1.0])) 
        fx2, fy2, fz2 = np.dot(self.KInv, np.array([fx2, fy2, 1.0]))  
        bx2, by2, bz2 = np.dot(self.KInv, np.array([bx2, by2, 1.0])) 
        
        pts = np.array([[fx1, bx1, fx2, bx2], [fy1, by1, fy2, by2],
                        [fz1, bz1, fz2, bz2]])
        
        scale = 3.0 / 2.0 - np.sqrt(5) / 2.0
        
        a = np.pi / 8.0
        b = 5.0 * np.pi / 8.0
        
        angs = np.array([a, a + scale * (b - a), b - scale * (b - a), b])        
        errs = np.array([0.0, 0.0, 0.0, 0.0])

        for ii in range(4):
            
            R = np.array([[1.0, 0.0, 0.0], 
                          [0.0, np.cos(angs[ii]), -np.sin(angs[ii])], 
                          [0.0, np.sin(angs[ii]), np.cos(angs[ii])]])
        
            newPts = np.dot(R, pts)
            newPts = newPts[0:2, :] / newPts[2,:]
            
            diffPts = np.array([newPts[:,1] - newPts[:, 0], 
                            newPts[:, 3] - newPts[:, 2]])
                            
            errs[ii] = (np.linalg.norm(diffPts[0,:]) - 
                        np.linalg.norm(diffPts[1,:]))**2
            
        
        for ii in range(15):
            
            if errs[1] < errs[2]:
                
                angs[2:4] = angs[1:3]
                errs[2:4] = errs[1:3]
                
                newIdx = 1
                
            else:
                
                angs[0:2] = angs[1:3]
                errs[0:2] = errs[1:3]
                
                newIdx = 2
                
            angs[3 - newIdx : 5 - newIdx] = angs[3 - newIdx : 5 - newIdx]
            errs[3 - newIdx : 5 - newIdx] = errs[3 - newIdx : 5 - newIdx]
            
            angs[newIdx] = angs[3 * newIdx - 3] + scale * (
                angs[-3 * newIdx + 6] - angs[3 * newIdx - 3])
                
            errs[newIdx] = self.rotationErr(pts, angs[newIdx])
            
        Hangle = np.sum(angs[1:3]) / 2.0    
            
        print pts
        print Hangle * 180 / np.pi
                
        self.H = np.array([[1.0, 0.0, 0.0],
                           [0.0, np.cos(Hangle), -np.sin(Hangle)],
                           [0.0, np.sin(Hangle), np.cos(Hangle)]])
                                      
        self.H = np.dot(self.H, self.KInv)
                           
                           
        
    def rotationErr(self, pts, ang):
        
        R = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ang), -np.sin(ang)], 
                      [0.0, np.sin(ang), np.cos(ang)]])
        
        newPts = np.dot(R, pts)
        newPts = newPts[0:2, :] / newPts[2,:]
        
        diffPts = np.array([newPts[:,1] - newPts[:, 0], 
                            newPts[:, 3] - newPts[:, 2]])
                            
        errs = (np.linalg.norm(diffPts[0,:]) - np.linalg.norm(diffPts[1,:]))**2
        
        return errs
        
    
    def testMoveCar(self):
        
        for ii in range(2):
            
            for jj in range(3):
                
                print ii, jj
                temp.moveCar(ii + 1, jj)
                time.sleep(1)
                
        
        
if __name__ == '__main__':

    temp = ChaseBall()

    doTest = 0
    
    if doTest :
        
        temp.testMoveCar()
        
    else:
  
        temp.setImgScale()
        
        for ii in range(3):
        
            temp.addColor(ii)
        
        
        temp.estimateHomography()
#    
        while True:
#        for ii in range(5):
        
           forward, turn = temp.determineMotion()
#    
#       print forward, turn
           temp.moveCar(forward, turn)
#        
           cv2.imshow('ThresholdedImage', temp.displayImg)
           cv2.waitKey()
           time.sleep(0.5)
    
    
#    temp.addColor(0)
    
