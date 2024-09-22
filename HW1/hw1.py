import cv2
import math
import numpy as np
import torch
import sys
from PIL import ImageTk, Image
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
from torchsummary import summary
import torchvision.models as models
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLineEdit,
                             QLabel, QPushButton, QMainWindow,  QFileDialog)
from PyQt5.QtGui import QPixmap, QImage
from torchvision import models
from torchvision.transforms import v2
from torchvision import transforms
import matplotlib.pyplot as plt


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Gui(QtWidgets.QMainWindow):
    ga_img =[]
    bi_img =[]
    me_img =[]
    sobel_x = [[-1,0,1],
               [-2,0,2],
               [-1,0,1]]

    sobel_y = [[-1,-2,-1],
              [0,0,0],
              [1,2,1]]
    f_name = []

    def __init__(self, model_path):
        super().__init__()

        self.image_path = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model(model_path)


        self.initUI()

    def initUI(self):


        self.setWindowTitle("Hw1")
        self.setGeometry(100, 100, 1280, 720)
        self.setFixedSize(1280, 720)

        self.zone1 = QtWidgets.QGroupBox("1. Image Processing", self)
        self.zone1.setFont(QtGui.QFont("Arial", 13))
        self.zone1.setGeometry(100, 20, 200, 150)

        self.zone2 = QtWidgets.QGroupBox("2. Image Smoothing", self)
        self.zone2.setFont(QtGui.QFont("Arial", 13))
        self.zone2.setGeometry(100, 350, 200, 150)

        self.zone3 = QtWidgets.QGroupBox("3. Image Transformations", self)
        self.zone3.setFont(QtGui.QFont("Arial", 13))
        self.zone3.setGeometry(300, 20, 200, 200)

        self.zone4 = QtWidgets.QGroupBox("4. Transforms", self)
        self.zone4.setFont(QtGui.QFont("Arial", 13))
        self.zone4.setGeometry(300, 350, 300, 250)

        self.zone5 = QtWidgets.QGroupBox("5. VGG19", self)
        self.zone5.setFont(QtGui.QFont("Arial", 13))
        self.zone5.setGeometry(500, 20, 400, 250)

        Color_Separation = QPushButton("1.1 Color Separation", self)
        Color_Separation.setGeometry(120, 40, 150, 30)
        Color_Separation.clicked.connect(self.Color_Separation)

        Color_Transformation = QPushButton("1.2 Color Transformation", self)
        Color_Transformation.setGeometry(120, 80, 150, 30)
        Color_Transformation.clicked.connect(self.Color_Transformation)

        Color_Extraction = QPushButton("1.3 Color Extraction", self)
        Color_Extraction.setGeometry(120, 120, 150, 30)
        Color_Extraction.clicked.connect(self.Color_Extraction)


        Gaussian_blur = QPushButton("2.1 Gaussian blur", self)
        Gaussian_blur.setGeometry(120, 370, 150, 30)
        Gaussian_blur.clicked.connect(self.Gaussian_blur)

        Load_Image = QPushButton("Load Image", self)
        Load_Image.setGeometry(120, 200, 150, 30)
        Load_Image.clicked.connect(self.Load_Image)

        
        Bilateral_filter = QPushButton("2.2 Bilateral filter", self)
        Bilateral_filter.setGeometry(120, 410, 150, 30)
        Bilateral_filter.clicked.connect(self.Bilateral_filter)


        Median_filter = QPushButton("2.3 Median filter", self)
        Median_filter.setGeometry(120, 450, 150, 30)
        Median_filter.clicked.connect(self.Median_filter)


        Sobel_X = QPushButton("3.1 Sobel X", self)
        Sobel_X.setGeometry(320, 40, 150, 30)
        Sobel_X.clicked.connect(self.Sobel_X)

        Sobel_Y = QPushButton("3.2 Sobel Y", self)
        Sobel_Y.setGeometry(320, 80, 150, 30)
        Sobel_Y.clicked.connect(self.Sobel_Y)

        Combination_and_Threshold = QPushButton("3.3 Combination and Threshold", self)
        Combination_and_Threshold.setGeometry(320, 120, 160, 30)
        Combination_and_Threshold.clicked.connect(self.Combination_and_Threshold)

        Gradient_Angle = QPushButton("3.4 Gradient Angle", self)
        Gradient_Angle.setGeometry(320, 160, 150, 30)
        Gradient_Angle.clicked.connect(self.Gradient_Angle)

 

        s = QLabel('Scaling:', self)

        r = QLabel('Rotation:', self)

        tx = QLabel('Tx:', self)

        ty = QLabel('Ty:', self)

        global mylineedit_s
        mylineedit_s = QLineEdit(self)
        global mylineedit_r
        mylineedit_r = QLineEdit(self)
        global mylineedit_tx
        mylineedit_tx = QLineEdit(self)
        global mylineedit_ty
        mylineedit_ty = QLineEdit(self)



        deg = QLabel('deg', self)
        
        pixelx = QLabel('pixel', self)
        
        pixely = QLabel('pixel', self)
        

        s.setGeometry(320, 430, 150, 30)
        
        r.setGeometry(320, 390, 150, 30)
        deg.setGeometry(530, 380, 160, 50)
        tx.setGeometry(320, 470, 150, 30)
        pixelx.setGeometry(530, 460, 160, 50)   
        ty.setGeometry(320, 500, 150, 30)
        pixely.setGeometry(530, 500, 160, 50)
        



        mylineedit_s.setGeometry(360, 430, 150, 30)
        mylineedit_r.setGeometry(360, 390, 150, 30)
        mylineedit_tx.setGeometry(360, 470, 150, 30)
        mylineedit_ty.setGeometry(360, 510, 150, 30)









        Transforms = QPushButton("4. Transforms", self)
        Transforms.setGeometry(320, 550, 150, 30)
        Transforms.clicked.connect(self.Transforms)


        Show_Augmented_Images = QPushButton("1. Show Augmented Images", self)
        Show_Augmented_Images.setGeometry(520, 80, 150, 30)
        Show_Augmented_Images.clicked.connect(self.Show_Augmented_Images)


        Show_Model_Structure = QPushButton("2. Show Model Structure", self)
        Show_Model_Structure.setGeometry(520, 120, 150, 30)
        Show_Model_Structure.clicked.connect(self.Show_Model_Structure)


        Show_Accuracy_and_Loss = QPushButton("3. Show Accuracy and Loss", self)
        Show_Accuracy_and_Loss.setGeometry(520, 160, 150, 30)
        Show_Accuracy_and_Loss.clicked.connect(self.Show_Accuracy_and_Loss)


        Inference = QPushButton("4. Inference", self)
        Inference.setGeometry(520, 200, 200, 30)
        Inference.clicked.connect(self.Inference)

        Load_img = QPushButton("Load Image", self)
        Load_img.setGeometry(520, 40, 150, 30)
        Load_img.clicked.connect(self.Load_img)

        Pre = QLabel('Predict=', self)
        Pre.setGeometry(730, 40, 160, 50)

        self.area6 = QtWidgets.QGroupBox("", self)
        self.area6.setGeometry(730, 90, 128, 128)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(730, 50, 128, 128)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(530, 240, 400, 30)      
        

    def Load_Image(self):

        try:
             file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
             self.f_name = file_name[0]

        except:
            pass

    def Color_Separation(self):

        img = cv2.imread(self.f_name)
        b, g, r =cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype = "uint8")
        merged_r = cv2.merge([zeros,zeros,r])
        merged_g = cv2.merge([zeros,g,zeros])
        merged_b = cv2.merge([b,zeros,zeros])
        
        cv2.imshow("R channel",merged_r)
        cv2.imshow("G channel",merged_g)
        cv2.imshow("B channel",merged_b)
        
       
        
        cv2.waitKey(0)
        #cv2.destoryAllWindows()
        
    def Color_Transformation(self):
        img = cv2.imread(self.f_name)
        b, g, r =cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype = "uint8")
        merged_r = cv2.merge([zeros,zeros,r])
        merged_g = cv2.merge([zeros,g,zeros])
        merged_b = cv2.merge([b,zeros,zeros])
        merged_bl = cv2.merge([zeros,zeros,zeros])
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
         
        h, w, c = img.shape
        dst = np.zeros((h,w), dtype=np.uint8)
        for row in range(h):
                for col in range(w):
                        b, g, r = np.int32(img[row,col])
                        
                        y = ((b+g+r)%255)/3
                        dst[row, col] = y
        
        
        cv2.imshow("I1",gray)
        cv2.imshow("I2",dst)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()
        
    def Color_Extraction(self):
        img = cv2.imread(self.f_name)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_y = cv2.inRange(hsv_img, (11, 43, 25), (34, 255, 255))
        mask_g = cv2.inRange(hsv_img, (35, 43, 25), (99, 255, 255))
        
        mask = cv2.add(mask_y, mask_g)
        mask_bgr= cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_img = cv2.bitwise_not(mask_bgr,img,mask)
        
        
        cv2.imshow("I1",mask)
        cv2.imshow("I2",mask_img)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()
        
        
    def Gaussian_blur(self):
    
        
        self.ga_img = cv2.imread(self.f_name)
        cv2.namedWindow('I1')
        cv2.createTrackbar('magnitude', 'I1', 1, 5, self.updateGaussian)
        
        
        cv2.imshow('I1',self.ga_img)
       
        
        cv2.waitKey(0)
        #cv2.destoryAllWindows()


        
        
    def updateGaussian(self,x):
        m = cv2.getTrackbarPos('magnitude', 'I1')
        new_ga_img = self.ga_img
        new_ga_img = cv2.GaussianBlur(new_ga_img, (2*m+1, 2*m+1), 0)
        cv2.imshow('I1',new_ga_img)
     
    def Bilateral_filter(self):
    
        
        self.bi_img = cv2.imread(self.f_name)
        cv2.namedWindow('I2')
        cv2.createTrackbar('magnitude', 'I2', 1, 5, self.updateBilateral)
        
        
        cv2.imshow('I2',self.bi_img)
       
        
        cv2.waitKey(0)
        #cv2.destoryAllWindows()


        
        
    def updateBilateral(self,x):
        m = cv2.getTrackbarPos('magnitude', 'I2')
        new_bi_img = self.bi_img
        new_bi_img = cv2.bilateralFilter(new_bi_img, 2*m+1,90,90)
        cv2.imshow('I2',new_bi_img)
        
    def Median_filter(self):
    
        
        self.me_img = cv2.imread(self.f_name)
        cv2.namedWindow('I3')
        cv2.createTrackbar('magnitude', 'I3', 1, 5, self.updateMedian)
        
        
        cv2.imshow('I3',self.me_img)
       
        
        cv2.waitKey(0)
        #cv2.destoryAllWindows()


        
        
    def updateMedian(self,x):
        m = cv2.getTrackbarPos('magnitude', 'I3')
        new_me_img = self.me_img
        new_me_img = cv2.medianBlur(new_me_img, 2*m+1)
        cv2.imshow('I3',new_me_img)
        
    def Sobel_X(self):
        img = cv2.imread(self.f_name)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        h, w, c = img.shape

        
        dst = np.zeros((h-2,w-2), dtype=np.int32)
        dst2 = np.zeros((h-2,w-2), dtype=np.uint8)
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst[row][col] = abs(self.sobel_x[0][0]*new_gray[row-1][col-1]+self.sobel_x[0][1]*new_gray[row-1][col]+self.sobel_x[0][2]*new_gray[row-1][col+1]+self.sobel_x[1][0]*new_gray[row][col-1]+self.sobel_x[1][1]*new_gray[row][col]+self.sobel_x[1][2]*new_gray[row][col+1]+self.sobel_x[2][0]*new_gray[row+1][col-1]+self.sobel_x[2][1]*new_gray[row+1][col]+self.sobel_x[2][2]*new_gray[row+1][col+1])
                             
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst[row][col] = abs(dst[row][col])                      
        
        Max=-1000000
        Min=1000000
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if Max < dst[row][col]:
                      Max = dst[row][col]
                     if Min > dst[row][col]:
                      Min = dst[row][col]
                      
                      
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                
                     dst[row][col] = (dst[row][col]-Min)*255/(Max-Min)
                     
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                   dst2[row][col]=dst[row][col]
                   
        
                     


               
        cv2.imshow("I1",dst2)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()

    def Sobel_Y(self):
        img = cv2.imread(self.f_name)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        h, w, c = img.shape

        
        dst = np.zeros((h-2,w-2), dtype=np.int32)
        dst2 = np.zeros((h-2,w-2), dtype=np.uint8)
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst[row][col] = abs(self.sobel_y[0][0]*new_gray[row-1][col-1]+self.sobel_y[0][1]*new_gray[row-1][col]+self.sobel_y[0][2]*new_gray[row-1][col+1]+self.sobel_y[1][0]*new_gray[row][col-1]+self.sobel_y[1][1]*new_gray[row][col]+self.sobel_y[1][2]*new_gray[row][col+1]+self.sobel_y[2][0]*new_gray[row+1][col-1]+self.sobel_y[2][1]*new_gray[row+1][col]+self.sobel_y[2][2]*new_gray[row+1][col+1])
                             
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst[row][col] = abs(dst[row][col])                      
        
        Max=-1000000
        Min=1000000
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if Max < dst[row][col]:
                      Max = dst[row][col]
                     if Min > dst[row][col]:
                      Min = dst[row][col]
                      
                      
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                
                     dst[row][col] = (dst[row][col]-Min)*255/(Max-Min)
                     
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                   dst2[row][col]=dst[row][col]
                   
        
                     


               
        cv2.imshow("I1",dst2)
        
        cv2.waitKey(0)
        #cv2.destoryAllWindows()  
         
    def Combination_and_Threshold(self):
        img = cv2.imread(self.f_name)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        h, w, c = img.shape

        
        dst = np.zeros((h-2,w-2), dtype=np.int32)
        dst2 = np.zeros((h-2,w-2), dtype=np.int32)
        
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst2[row][col] = self.sobel_x[0][0]*new_gray[row-1][col-1]+self.sobel_x[0][1]*new_gray[row-1][col]+self.sobel_x[0][2]*new_gray[row-1][col+1]+self.sobel_x[1][0]*new_gray[row][col-1]+self.sobel_x[1][1]*new_gray[row][col]+self.sobel_x[1][2]*new_gray[row][col+1]+self.sobel_x[2][0]*new_gray[row+1][col-1]+self.sobel_x[2][1]*new_gray[row+1][col]+self.sobel_x[2][2]*new_gray[row+1][col+1]



        dst3 = np.zeros((h-2,w-2), dtype=np.int32)
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst3[row][col] = self.sobel_y[0][0]*new_gray[row-1][col-1]+self.sobel_y[0][1]*new_gray[row-1][col]+self.sobel_y[0][2]*new_gray[row-1][col+1]+self.sobel_y[1][0]*new_gray[row][col-1]+self.sobel_y[1][1]*new_gray[row][col]+self.sobel_y[1][2]*new_gray[row][col+1]+self.sobel_y[2][0]*new_gray[row+1][col-1]+self.sobel_y[2][1]*new_gray[row+1][col]+self.sobel_y[2][2]*new_gray[row+1][col+1]

               
        dst4 = np.zeros((h-2,w-2), dtype=np.uint8)   
        dst5 = np.zeros((h-2,w-2), dtype=np.uint8)


        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst[row][col] = (dst3[row][col]**2+dst2[row][col]**2)**0.5
                     
        Max=-1000000
        Min=1000000
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if Max < dst[row][col]:
                      Max = dst[row][col]
                     if Min > dst[row][col]:
                      Min = dst[row][col]        
                  
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                   dst4[row][col]=(dst[row][col]-Min)*255/(Max-Min)
                   if dst4[row][col]<128:
                      dst5[row][col] = 0
                   else:
                      dst5[row][col] = 255 
               
        cv2.imshow("I1",dst4)
        cv2.imshow("I2",dst5)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()  
        
    def Gradient_Angle(self):
        img = cv2.imread(self.f_name)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        new_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        h, w, c = img.shape

        
        dst = np.zeros((h-2,w-2), dtype=np.int32)

        dst8 = np.zeros((h-2,w-2), dtype=np.int32)
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst8[row][col] = (self.sobel_x[0][0]*new_gray[row-1][col-1]+self.sobel_x[0][1]*new_gray[row-1][col]+self.sobel_x[0][2]*new_gray[row-1][col+1]+self.sobel_x[1][0]*new_gray[row][col-1]+self.sobel_x[1][1]*new_gray[row][col]+self.sobel_x[1][2]*new_gray[row][col+1]+self.sobel_x[2][0]*new_gray[row+1][col-1]+self.sobel_x[2][1]*new_gray[row+1][col]+self.sobel_x[2][2]*new_gray[row+1][col+1])

                             
                     


        dst9 = np.zeros((h-2,w-2), dtype=np.int32)
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst9[row][col] = (self.sobel_y[0][0]*new_gray[row-1][col-1]+self.sobel_y[0][1]*new_gray[row-1][col]+self.sobel_y[0][2]*new_gray[row-1][col+1]+self.sobel_y[1][0]*new_gray[row][col-1]+self.sobel_y[1][1]*new_gray[row][col]+self.sobel_y[1][2]*new_gray[row][col+1]+self.sobel_y[2][0]*new_gray[row+1][col-1]+self.sobel_y[2][1]*new_gray[row+1][col]+self.sobel_y[2][2]*new_gray[row+1][col+1])

                             
            
        dst4 = np.zeros((h-2,w-2), dtype=np.uint8)   


        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     dst[row][col] = (dst8[row][col]**2+dst9[row][col]**2)**0.5
                     
        Max=-1000000
        Min=1000000
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if Max < dst[row][col]:
                      Max = dst[row][col]
                     if Min > dst[row][col]:
                      Min = dst[row][col]        
                  
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                   dst4[row][col]=(dst[row][col]-Min)*255/(Max-Min)
                   
                   
        dst6 = np.zeros((h-2,w-2), dtype=np.uint8)   
        dst7 = np.zeros((h-2,w-2), dtype=np.uint8)
        
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                  if dst8[row][col] == 0 and dst9[row][col] <0:
                    cos = 270
                  elif dst8[row][col] == 0 and dst9[row][col] >0:
                    cos = 90
                  elif dst8[row][col] == 0 and dst9[row][col] ==0:
                    cos = 0
                  else:
                    cos = math.atan2(dst9[row][col],dst8[row][col])
                    cos*=57.3

                 
                  if dst8[row][col]<0 and dst9[row][col]>0:
                      cos+=180
                  if dst8[row][col]<0 and dst9[row][col]<0:
                      cos+=360
                  if dst8[row][col]>0 and dst9[row][col]<0:
                      cos+=180
                  
                  if cos>=120 and cos<=180:
                      dst6[row][col]=255
                  else:
                      dst6[row][col]=0
                      
                  if cos>=210 and cos<=330:
                      dst7[row][col]=255
                  else:
                      dst7[row][col]=0  
          
        dst6 = cv2.bitwise_and(dst4, dst6)
        dst7 = cv2.bitwise_and(dst4, dst7)
        
        
        cv2.imshow("I1",dst6)
        cv2.imshow("I2",dst7)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()  
    def Transforms(self):

        img = cv2.imread(self.f_name)
        rows, cols, c = img.shape


        M = cv2.getRotationMatrix2D((240,200),float(mylineedit_r.text()),float(mylineedit_s.text()))

        res = cv2.warpAffine(img,M,(cols,rows))
        H = np.float32([[1,0,0],[0,1,0]])

        H[0][2]= float(mylineedit_tx.text())
        H[1][2]= float(mylineedit_ty.text())
        res = cv2.warpAffine(res,H,(cols,rows))

        cv2.imshow("I1",res)
        cv2.waitKey(0)
        

        
        
    def Show_Augmented_Images(self):

        img_1 = Image.open(r"Q5_1/automobile.png")  
        img_2 = Image.open(r"Q5_1/bird.png")  
        img_3 = Image.open(r"Q5_1/cat.png")  
        img_4 = Image.open(r"Q5_1/deer.png")  
        img_5 = Image.open(r"Q5_1/dog.png")  
        img_6 = Image.open(r"Q5_1/frog.png")  
        img_7 = Image.open(r"Q5_1/horse.png")  
        img_8 = Image.open(r"Q5_1/ship.png")  
        img_9 = Image.open(r"Q5_1/truck.png")  
        H, W = 32, 32
        #transform = v2.Compose([
    #v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #v2.RandomCrop(32, padding=4),
    #v2.ToTensor(),
    #v2.RandomRotation(30),
   #v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
#])
        transform = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
   v2.RandomRotation(30)
])
        img_1 = transform(img_1)
        img_2 = transform(img_2)
        img_3 = transform(img_3)
        img_4 = transform(img_4)
        img_5 = transform(img_5)
        img_6 = transform(img_6)
        img_7 = transform(img_7)
        img_8 = transform(img_8)
        img_9 = transform(img_9)
        
        

        plt.subplot(3,3,1)
        plt.title('automobile')
        plt.imshow(img_1)
       
        
        plt.subplot(3,3,2)
        plt.title('bird')
        plt.imshow(img_2)
        
        plt.subplot(3,3,3)
        plt.title('cat')
        plt.imshow(img_3)
        
        plt.subplot(3,3,4)
        plt.title('deer')
        plt.imshow(img_4)
        
        plt.subplot(3,3,5)
        plt.title('dog')
        plt.imshow(img_5)
        
        plt.subplot(3,3,6)
        plt.title('frog')
        plt.imshow(img_6)
        
        plt.subplot(3,3,7)
        plt.title('horse')
        plt.imshow(img_7)
        
        plt.subplot(3,3,8)
        plt.title('ship')
        plt.imshow(img_8)
        
        plt.subplot(3,3,9)
        plt.title('truck')
        plt.imshow(img_9)
 	
 
        plt.show()
        cv2.waitKey(0)
        
    def Show_Model_Structure(self):
        device = torch.device("cuda")
        vgg19_bn = models.vgg19_bn(num_classes=10)
        vgg19_bn.to(device)
        summary(vgg19_bn, (3, 32, 32))
        cv2.waitKey(0)

    def Show_Accuracy_and_Loss(self):
        img = cv2.imread('training_curve.png')
        cv2.imshow("I1",img)
        cv2.waitKey(0)

    def load_model(self, model_path):
        try:
            self.model = models.vgg19_bn(num_classes=10)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading the model: {e}")
            
    def Inference(self):
        if hasattr(self, 'image_path') and self.model is not None:
            try:
                transform = v2.Compose([
                    v2.ToTensor(),
                    v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
                ])
                
                image = Image.open(self.image_path)
                transformed_image = transform(image)

                transformed_image = transformed_image.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model(transformed_image)

                _, predicted = torch.max(output, 1)

                self.result_label.setText(f"Predicted Class: {classes[predicted.item()]}")

                # Plot probability distribution using a histogram
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
                plt.bar(classes, probabilities[0])
                plt.xlabel("Class")
                plt.ylabel("Probability")
                plt.title("Probability Distribution")
                plt.show()
            except Exception as e:
                print(f"Error during inference: {e}")
        
    def Load_img(self):

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.jpg *.jpeg *.png *.bmp *.gif *.tif *.tiff);;All Files (*)", options=options)

        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            pixmap = pixmap.scaled(128, 128)
            self.image_label.setGeometry(730, 90,128,128)
            self.image_label.setPixmap(pixmap)

    def QImageToTensor(self, qimage):
        width, height = qimage.width(), qimage.height()
        ptr = qimage.bits()
        ptr.setsize(3 * width * height)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))
        arr = arr.transpose(2, 0, 1)  # 调整通道顺序
        tensor = torch.from_numpy(arr).float()
        tensor /= 255.0  # 标准化像素值
        return tensor
    
def main():
    app = QApplication(sys.argv)
    model_path = "best_model.pth"  # 请替换为您的模型文件路径
    mainWindow = Gui(model_path)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
