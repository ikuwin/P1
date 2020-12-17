import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

class PCAFaceRecognition:
	def __init__(self,components,N,M):
		self.componenets=components
		self.N=N
		self.M=M

	def main(self,trainPath):
		imgsVec = self.prepareImages(trainPath)
		meanImg = np.mean(imgsVec,1,keepdims=True)
		A = imgsVec-meanImg
		L = np.matmul(np.transpose(A),A)
		print('Finding eigenVec')
		w, v = np.linalg.eig(L)
		u = np.matmul(A,v)
		norm = np.linalg.norm(u)
		U = u/norm
		X=np.vstack((w,U))
		X=X[:, X[0].argsort()]
		w=X[0]
		U=X[1:,:]
		weights=np.matmul(np.transpose(U),A)
		return weights


	def prepareImages(self,trainPath):
		imageVmatrix=np.zeros((self.N*self.N,self.M))
		i=0
		for file in os.listdir(trainPath):
			img=cv2.imread(os.path.join(trainPath,file),0)
			img=cv2.resize(img,(self.N,self.N))
			imageVmatrix[:,i]=img.ravel()
			i+=1
		return imageVmatrix


PCAClass=PCAFaceRecognition(1,100,15)
weights=PCAClass.main("F:\TestProject\PCAIMG")
print(weights.shape)
print(weights[:,1])
