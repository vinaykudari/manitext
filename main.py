import argparse
import cv2
import json
from os import walk
import numpy as np

from manitext import ManiText
from nlp import Similarity

class ManiTextProcessor:
    def __init__(self, dataPath, outputPath):
        self.maniText = ManiText()
        self.dataPath = dataPath
        self.outputPath = outputPath
        
    @staticmethod
    def getAllFiles(path):
        for (dirPath, _, fileNames) in walk(path):
            for fileName in fileNames:
                yield dirPath, fileName
    
    def replaceTitleBgAll(self, cropFooter=True):
        mediaPath = f'{self.dataPath}/media'
        infoPath = f'{self.dataPath}/info.json'
        
        with open(infoPath, 'r') as f:
            info = json.load(f)
        
        for dirPath, fileName in self.getAllFiles(mediaPath):
            imgPath = f'{dirPath}/{fileName}'
            img = cv2.imread(imgPath)
            name = fileName.split('.')[0]

            if cropFooter:
                img = img[:-100, :]
            
            if name in info:
                try:
                    self.maniText.replaceWithBackground(
                        img, info[name]['title'],
                        Similarity.wordMatch, 0.70)
                except Exception as e:
                    print(f'Failed: {name}; E: {e}')
                
                cv2.imwrite(f'{self.outputPath}/_{fileName}', img)
                print('.', end='')
                
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', help='Path to data')
    parser.add_argument('-o', help='Path to output directory')
    args = parser.parse_args()
    
    dataPath, outputPath = args.d, args.o
    
    if dataPath and dataPath[-1] == '/':
        dataPath = dataPath[:-1]
        
    if outputPath and outputPath[-1] == '/':
        outputPath = outputPath[:-1]
    
    processor = ManiTextProcessor(dataPath, outputPath)
    processor.replaceTitleBgAll()
    
                
                
    