import cv2
import easyocr
import numexpr as ne
import numpy as np


class ManiText:
    def __init__(self, languages=['en'], gpu=True):
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    @staticmethod
    def binarizeImage(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        binarized = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 85, 11)
        
        return binarized
    
    def getTextBlobs(self, img):
        result = self.reader.readtext(img, paragraph=False)
        textBoxes = []
        
        for detection in result:
            topLeft = int(detection[0][0][0]), int(detection[0][0][1])
            bottomRight = int(detection[0][2][0]), int(detection[0][2][1])
            box = topLeft, bottomRight
            textBoxes.append([detection[1], box])
            
        return textBoxes

    @staticmethod
    def getBackgroundColor(img):
        img = img.reshape(-1, img.shape[-1])
        colRange = 256, 256, 256
        evalParams = {
            'a0': img[:, 0],'a1': img[:, 1],
            'a2': img[:, 2], 's0': colRange[0],
            's1': colRange[1]}
        a1D = ne.evaluate('a0*s0*s1+a1*s0+a2', evalParams)
        arr = np.array(np.unravel_index(np.bincount(a1D).argmax(), colRange))

        return arr.tolist()

    @staticmethod
    def drawRectangle(img, box, color):
        im = cv2.rectangle(img, box[0], box[1], color, -1)
        return im
    
    @staticmethod
    def getBoxRegion(img, box):
        (x1, y1), (x2, y2) = box
        h = y2 - y1
        w = x2 - x1
        return img[y1:y1+h, x1:x1+w]
    
    def replaceWithBackground(self, img, textToRemove, similarity, thresh):
        binImg = self.binarizeImage(img)
        textBoxes = self.getTextBlobs(binImg)
        
        for text, box in textBoxes:
            if similarity(text, textToRemove) > thresh:
                boxRegion = self.getBoxRegion(img, box)
                backgroundColor = self.getBackgroundColor(boxRegion)
                if img is not None:
                    cv2.rectangle(img, box[0], box[1], backgroundColor, -1)
                
        return img