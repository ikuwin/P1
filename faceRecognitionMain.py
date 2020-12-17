#from mtcnn.mtcnn import MTCNN
# import cv2
# import os
# import numpy as np
# from skimage import feature
# #from lbpFace import LBP
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# import random
#from MyLBP import customMe
# #from otherLBP import AkLBP
# from ldgp import Calculator
# import time
# from LBPClass import customMe
#
# class FaceRecognition():
#     def __init__(self,numPoints=24,radius=8):
#         #self.detector = MTCNN()
#         self.numPoints=numPoints
#         self.radius=radius
#
#
#
#     def showFeed(self):
#         cap=cv2.VideoCapture(0)
#         while True:
#             ret, img = cap.read()
#             #imgContainingFace = self.detect_face(img)['imgContainingFace']
#             #cv2.imshow('frame', imgContainingFace)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cap.release()
#         cv2.destroyAllWindows()
#
#
#     # def detect_face(self,img):
#     #     outputDetectFace={}
#     #     #foundFace=False
#     #     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     #     faces = self.detector.detect_faces(imgRGB)
#     #     for face in faces:
#     #         x1, y1, w1, h1 = face['box']
#     #         cv2.rectangle(imgRGB, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 3)
#     #         boundingValues=(x1,y1,w1,h1)
#     #         outputDetectFace.update({'boundingValues': boundingValues})
#     #     imgfinal = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
#     #     outputDetectFace.update({'imgContainingFace':imgfinal})
#     #     return outputDetectFace
#
#     def trainRecognizer(self,trainPath,printAcc,testPath=None):
#         desc = Calculator(1,1,16)
#         data = []
#         labels = []
#         print('Extracting Features.....')
#         for imageFolder in os.listdir(trainPath):
#             imagePath = os.path.join(trainPath, imageFolder)
#             for trainImg in os.listdir(imagePath):
#                 if trainImg == "Thumbs.db":
#                     continue
#                 image = cv2.imread(os.path.join(imagePath, trainImg))
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 tic=time.time()
#                 hist = desc.calc_hist(gray[50:-50,50:-50])
#                 toc=time.time()
#                 #print(toc-tic)
#                 #(hist, _) = np.histogram(hist.ravel())
#                 # = hist.astype("float")
#                 #hist /= (hist.sum())
#                 labels.append(int(imageFolder[-1]))
#                 data.append(hist)
#             print("Processed folder " + imageFolder)
#             ###################################
#             if imageFolder=="Subject05":
#                 break
#         print('Completed Feature Extraction!')
#         print('Training Classifier.......')
#         temp = list(zip(data, labels))
#         random.shuffle(temp)
#         data, labels = zip(*temp)
#         model = KNeighborsClassifier()
#         model.fit(data, labels)
#         print('Done Training')
#         if printAcc:
#             correct = 0
#             total = 0
#             print('Calculating accuracy....')
#             for imageFolder in os.listdir(testPath):
#                 imagePath = os.path.join(testPath, imageFolder)
#                 for testImg in os.listdir(imagePath):
#                     if testImg=="Thumbs.db":
#                         continue
#                     image = cv2.imread(os.path.join(imagePath, testImg))
#                     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                     hist = desc.calc_hist(gray[50:-50,50:-50])
#                     #(hist, _) = np.histogram(hist.ravel())
#                     #hist = hist.astype("float")
#                     #hist /= (hist.sum())
#                     prediction = model.predict(hist.reshape(1,-1))
#                     total += 1
#                     if prediction == int(imageFolder[-1]):
#                         correct += 1
#                 print("Done "+imageFolder)
#                 tempacc=(correct/total)*100
#                 print("Total : "+str(total)+" Correct : "+str(correct)+" Accuracy : "+str(tempacc))
#                 #######################################
#                 if imageFolder == "Subject05":
#                     break
#             acc=(correct/total)*100
#             print('Accuray on test set is : '+str(acc)+'%')
#         return model

    # def recognizeFace(self,img,model):
    #     desc = LBP(self.numPoints, self.radius)
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     hist = desc.describe(gray)
    #     prediction = model.predict(hist.reshape(1, -1))
    #     cv2.putText(img,str(prediction),(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)
    #     return img


import cv2
import numpy as np
import os
import random
from sklearn.neighbors import KNeighborsClassifier
#from ldgp import Calculator
from LBPClass import customMe
import time


class FaceRecognition:
    def __init__(self, numPoints=24, radius=8):
        self.numPoints = numPoints
        self.radius = radius


    def trainRecognizer(self, trainPath, printAcc, testPath = None):
        ccc=0
        desc = customMe(16)
        data = []
        labels = []
        print('Extracting Features.....')
        for imageFolder in os.listdir(trainPath):
            imagePath = os.path.join(trainPath, imageFolder)
            for trainImg in os.listdir(imagePath):
                if trainImg == "Thumbs.db":
                    continue
                image = cv2.imread(os.path.join(imagePath, trainImg))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                tic=time.time()
                hist = desc.findLBP(gray[40:-70,120:-180])
                toc=time.time()
                labels.append(int(imageFolder[-1]))
                data.append(hist)
                print(toc-tic)
                ccc+=1
            print("Processed folder " + imageFolder)
            if imageFolder == "Subject05":
                break
        print('Completed Feature Extraction!')
        print('Training Classifier.......')
        temp = list(zip(data, labels))
        random.shuffle(temp)
        data, labels = zip(*temp)
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(data, labels)
        print('Done Training')
        if printAcc:
            correct = 0
            total = 0
            print('Calculating accuracy....')
            for imageFolder in os.listdir(testPath):
                imagePath = os.path.join(testPath, imageFolder)
                for testImg in os.listdir(imagePath):
                    if testImg == "Thumbs.db":
                        continue
                    image = cv2.imread(os.path.join(imagePath, testImg))
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    hist = desc.findLBP(gray[40:-70,120:-180])
                    hist = np.array(hist)
                    prediction = model.predict(hist.reshape(1, -1))
                    total += 1
                    if prediction == int(imageFolder[-1]):
                        correct += 1
                print("Done "+imageFolder)
                tempacc = (correct/total)*100
                print("Total : "+str(total)+" Correct : "+str(correct)+" Accuracy : "+str(tempacc))
                if imageFolder == "Subject05":
                    break
            acc = (correct/total)*100
            print('Accuracy on test set is : '+str(acc)+'%')
        return model











