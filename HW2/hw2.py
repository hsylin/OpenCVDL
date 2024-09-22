import cv2
import math
import numpy as np
import torch
import sys
from PIL import ImageQt, Image
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib import pyplot as plt
from torchsummary import summary
import torchvision.models as models
from PyQt5.QtWidgets import (QApplication, QWidget, QGridLayout, QLineEdit,
                             QLabel, QPushButton, QMainWindow,  QFileDialog)
from PyQt5.QtWidgets import (QApplication, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QFile, QIODevice
from torchvision import models
from torchvision.transforms import v2
from torchvision import transforms
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic, QtWidgets
import torchvision
import os
import torch.nn as nn
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes_2 = ('Cat', 'Dog')
import os
from torch.utils.data import Dataset
from torchvision.io import read_image



class Gui(QtWidgets.QMainWindow):
    ga_img =[]
    bi_img =[]
    me_img =[]
    ele_structure = [[1,1,1],
               [1,1,1],
               [1,1,1]]

    f_name = []
    num=0
    model_path_load = []
    root_dir =[]
    label_dir_c=[]
    label_dir_d=[]
    data_dir_c=[]
    data_dir_d=[]
    img_names_c=[]
    img_names_d=[]
    
    def __init__(self,model_path_Vgg, model_path_res, parent=None):
        super(Gui, self).__init__(parent)
        
        self.setWindowTitle("Hw2")
        
        
        self.image_path = None
        self.model_Vgg = None
        self.model_res = None        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_model_Vgg(model_path_Vgg)
        self.load_model_res(model_path_res)        
        #self.model_path_load = model_path


        
        #实例化QPixmap类

        self.pix = QPixmap()
        #起点，终点
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        #初始化
        self.initUI()

     

    def initUI(self):

        # 画布大小为400*400，背景为白色
        self.pix = QPixmap(300, 300)
        self.pix.fill(Qt.black)
        
        self.setGeometry(100, 100, 1280, 720)
        self.resize(1280, 720)
        
        self.zone1 = QtWidgets.QGroupBox("1. Hough Circke Transform", self)
        self.zone1.setFont(QtGui.QFont("Arial", 13))
        self.zone1.setGeometry(100, 20, 200, 150)

        self.zone2 = QtWidgets.QGroupBox("2. Histogram Equalization", self)
        self.zone2.setFont(QtGui.QFont("Arial", 13))
        self.zone2.setGeometry(100, 350, 200, 150)

        self.zone3 = QtWidgets.QGroupBox("3. Morphology Operation", self)
        self.zone3.setFont(QtGui.QFont("Arial", 13))
        self.zone3.setGeometry(300, 20, 200, 200)

        self.zone4 = QtWidgets.QGroupBox("4. MNIST Classifier Using VGG19", self)
        self.zone4.setFont(QtGui.QFont("Arial", 13))
        self.zone4.setGeometry(300, 350, 700, 500)

        self.zone5 = QtWidgets.QGroupBox("5. ResNet50", self)
        self.zone5.setFont(QtGui.QFont("Arial", 13))
        self.zone5.setGeometry(500, 20, 400, 250)

        Draw_contour = QPushButton("1.1 Draw contour", self)
        Draw_contour.setGeometry(120, 40, 150, 30)
        Draw_contour.clicked.connect(self.Draw_contour)

        Count_Coins = QPushButton("1.2 Count Coins", self)
        Count_Coins.setGeometry(120, 80, 150, 30)
        Count_Coins.clicked.connect(self.Count_Coins)

        self.count = QLabel('There are _ coins in the image.', self)
        self.count.setGeometry(120, 110, 150, 30)


        Histogram_Equalization = QPushButton("2.1 Histogram Equalization", self)
        Histogram_Equalization.setGeometry(120, 370, 150, 30)
        Histogram_Equalization.clicked.connect(self.Histogram_Equalization)

        Load_Image = QPushButton("Load Image", self)
        Load_Image.setGeometry(120, 200, 150, 30)
        Load_Image.clicked.connect(self.Load_Image)
        
        


        Closing = QPushButton("3.1 Closing", self)
        Closing.setGeometry(320, 40, 150, 30)
        Closing.clicked.connect(self.Closing)

        Opening = QPushButton("3.2 Opening", self)
        Opening.setGeometry(320, 80, 150, 30)
        Opening.clicked.connect(self.Opening)


 

        Show_Model_Structure_VGG19 = QPushButton("4.1. Show Model Structure", self)
        Show_Model_Structure_VGG19.setGeometry(320, 390, 150, 30)
        Show_Model_Structure_VGG19.clicked.connect(self.Show_Model_Structure_VGG19)

        Show_Accuracy_and_Loss_VGG19 = QPushButton("4.2. Show Accuracy an Loss", self)
        Show_Accuracy_and_Loss_VGG19.setGeometry(320, 440, 150, 30)
        Show_Accuracy_and_Loss_VGG19.clicked.connect(self.Show_Accuracy_and_Loss_VGG19)

        Predict_VGG19 = QPushButton("4.3. Predict", self)
        Predict_VGG19.setGeometry(320, 510, 150, 30)
        Predict_VGG19.clicked.connect(self.Predict_VGG19)

        Reset_VGG19 = QPushButton("4.4. Reset", self)
        Reset_VGG19.setGeometry(320, 550, 150, 30)
        Reset_VGG19.clicked.connect(self.Reset_VGG19)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(320, 590, 400, 30) 




        Show_Images = QPushButton("5.1. Show Images", self)
        Show_Images.setGeometry(520, 80, 150, 30)
        Show_Images.clicked.connect(self.Show_Images)


        Show_Model_Structure = QPushButton("5.2. Show Model Structure", self)
        Show_Model_Structure.setGeometry(520, 120, 150, 30)
        Show_Model_Structure.clicked.connect(self.Show_Model_Structure)


        Show_Comparison = QPushButton("5.3. Show Comparison", self)
        Show_Comparison.setGeometry(520, 160, 150, 30)
        Show_Comparison.clicked.connect(self.Show_Comparison)


        inference_res = QPushButton("5.4. Inference", self)
        inference_res.setGeometry(520, 200, 200, 30)
        inference_res.clicked.connect(self.inference_res)

        Load_img = QPushButton("Load Image", self)
        Load_img.setGeometry(520, 40, 150, 30)
        Load_img.clicked.connect(self.Load_img)

        Pre = QLabel('Predict=', self)
        Pre.setGeometry(730, 40, 160, 50)

        self.area6 = QtWidgets.QGroupBox("", self)
        self.area6.setGeometry(730, 90, 128, 128)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(730, 50, 128, 128)

        self.result_label_2 = QLabel(self)
        self.result_label_2.setGeometry(780, 50, 400, 30) 
        
    
    def Load_Image(self):

        try:
             file_name = QFileDialog.getOpenFileName(self, 'open file', '.')
             self.f_name = file_name[0]

        except:
            pass

    def Draw_contour(self):

        img = cv2.imread(self.f_name)
        img2 = cv2.imread(self.f_name)
        img3 = cv2.imread(self.f_name)
        h, w, c = img.shape
        for row in range(0,(h)):
                for col in range(0,(w)):
                     img3[row][col][0]=0
                     img3[row][col][1]=0
                     img3[row][col][2]=0

        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,20,
        param1=50,param2=30,minRadius=20,maxRadius=40)
        circles = np.uint16(np.around(circles))
        
        for i in circles[0,:]:
         # draw the outer circle
           cv2.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
         # draw the center of the circle
           cv2.circle(img3,(i[0],i[1]),1,(255,255,255),2)
           self.num+=1
        
        
        cv2.imshow('Img_src',img)       
        cv2.imshow('Img_process',img2)
        cv2.imshow('Circle_center',img3)  
        cv2.waitKey(0)
        #cv2.destoryAllWindows()
        
    def Count_Coins(self):
        self.count.setText(f'There are {self.num} coins in the image.')
        self.num=0
        #cv2.destoryAllWindows()
        

        
    def Histogram_Equalization(self):
    

        img_src = cv2.imread(self.f_name)
        img_src = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)        
        img_dst = cv2.equalizeHist(img_src)
        

        hist_src = cv2.calcHist([img_src], [0], None, [256], [0, 256])
        hist_dst = cv2.calcHist([img_dst], [0], None, [256], [0, 256])

        #显示图像
        fig,ax = plt.subplots(2,3)
        ax[1,0].set_title('Historgram of original')
        ax[1,0].bar(range(256), hist_src.ravel())
        ax[1,0].set_xlabel('Gray scale')
        ax[1,0].set_ylabel('Frequency')
        ax[0,1].set_title('Equalized with OpenCV') 
        ax[0,1].imshow(cv2.cvtColor(img_dst,cv2.COLOR_BGR2RGB),'gray')
        
        
        ax[1,1].set_title('Historgram of Equalized(OpenCV)')
        ax[1,1].bar(range(256), hist_dst.ravel())
        ax[1,1].set_xlabel('Gray scale')
        ax[1,1].set_ylabel('Frequency')        
        ax[0,0].set_title('Original Image')
        ax[0,0].imshow(cv2.cvtColor( img_src, cv2.COLOR_BGR2RGB),'gray')


        
        # Load the image
    
        original_array = np.array(img_src)

        # Step 1: Calculate histogram using numpy.histogram()
        hist, bins = np.histogram(original_array.flatten(), bins=256, range=[0, 256])

        # Step 2: Calculate PDF from normalized histogram
        pdf = hist / np.sum(hist)

        # Step 3: Calculate CDF by cumulatively summing PDF
        cdf = np.cumsum(pdf)

        # Step 4: Create lookup table based on rounded CDF values
        lookup_table = np.round(cdf * 255).astype('uint8')

        # Step 5: Apply lookup table to original image
        equalized_array = lookup_table[original_array]

        # Step 6: Create new equalized image
        equalized_hist, _ = np.histogram(equalized_array.flatten(), bins=256, range=[0, 256])
        

        ax[1,2].set_title('Historgram of Equalized(Manual)')
        ax[1,2].bar(range(256), equalized_hist.ravel())
        ax[1,2].set_xlabel('Gray scale')
        ax[1,2].set_ylabel('Frequency')            
        ax[0,2].set_title('Equalized Manually')
        ax[0,2].imshow(equalized_array, cmap='gray')        


        plt.show()  
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
        
    def Closing(self):
        img = cv2.imread(self.f_name)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
        ret2 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
        ret3 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))        
        h, w= ret.shape

        


        for row in range(0,(h)):
                for col in range(0,(w)):
                     if(ret[row][col]<=127):
                         ret[row][col]=0
                     else:
                         ret[row][col]=255
                         
        dst = np.zeros((h-2,w-2), dtype=np.int8)       
        ans=-1000000                     
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if ans<ret[row-1][col-1]:
                        ans = ret[row-1][col-1]
                     if(ans<ret[row-1][col]):
                        ans = ret[row-1][col]
                     if(ans<ret[row-1][col+1]):
                        ans = ret[row-1][col+1]
                     if(ans<ret[row][col-1]):
                        ans = ret[row][col-1]
                     if(ans<ret[row][col]):
                        ans = ret[row][col]
                     if(ans<ret[row][col+1]):
                        ans = ret[row][col+1]
                     if(ans<ret[row+1][col-1]):
                        ans = ret[row+1][col-1]
                     if(ans<ret[row+1][col]):
                        ans = ret[row+1][col]
                     if(ans<ret[row+1][col+1]):
                        ans = ret[row+1][col+1]
                     ret2[row][col] =ans
                     ans=-1000000


        ans=1000000                     
        h, w= dst.shape
        dst2 = np.zeros((h-2,w-2), dtype=np.int8)
        
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if(ans>ret2[row-1][col-1]):
                        ans = ret2[row-1][col-1]
                     if(ans>ret2[row-1][col]):
                        ans = ret2[row-1][col]
                     if(ans>ret2[row-1][col+1]):
                        ans = ret2[row-1][col+1]
                     if(ans>ret2[row][col-1]):
                        ans = ret2[row][col-1]
                     if(ans>ret2[row][col]):
                        ans = ret2[row][col]
                     if(ans>ret2[row][col+1]):
                        ans = ret2[row][col+1]
                     if(ans>ret2[row+1][col-1]):
                        ans = ret2[row+1][col-1]
                     if(ans>ret2[row+1][col]):
                        ans = ret2[row+1][col]
                     if(ans>ret2[row+1][col+1]):
                        ans = ret2[row+1][col+1]
                     ret3[row][col] =ans                 
                     ans=1000000

               
        cv2.imshow("I1",ret3)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()

    def Opening(self):
        img = cv2.imread(self.f_name)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
        ret2 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))
        ret3 = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0))        
        h, w= ret.shape

        


        for row in range(0,(h)):
                for col in range(0,(w)):
                     if(ret[row][col]<=127):
                         ret[row][col]=0
                     else:
                         ret[row][col]=255
                         


        ans=1000000                     
        
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if(ans>ret[row-1][col-1]):
                        ans = ret[row-1][col-1]
                     if(ans>ret[row-1][col]):
                        ans = ret[row-1][col]
                     if(ans>ret[row-1][col+1]):
                        ans = ret[row-1][col+1]
                     if(ans>ret[row][col-1]):
                        ans = ret[row][col-1]
                     if(ans>ret[row][col]):
                        ans = ret[row][col]
                     if(ans>ret[row][col+1]):
                        ans = ret[row][col+1]
                     if(ans>ret[row+1][col-1]):
                        ans = ret[row+1][col-1]
                     if(ans>ret[row+1][col]):
                        ans = ret[row+1][col]
                     if(ans>ret[row+1][col+1]):
                        ans = ret[row+1][col+1]
                     ret2[row][col] =ans                 
                     ans=1000000
      
        ans=-1000000                     
        for row in range(1,(h-2)):
                for col in range(1,(w-2)):
                     if ans<ret2[row-1][col-1]:
                        ans = ret2[row-1][col-1]
                     if(ans<ret2[row-1][col]):
                        ans = ret2[row-1][col]
                     if(ans<ret2[row-1][col+1]):
                        ans = ret2[row-1][col+1]
                     if(ans<ret2[row][col-1]):
                        ans = ret2[row][col-1]
                     if(ans<ret2[row][col]):
                        ans = ret2[row][col]
                     if(ans<ret2[row][col+1]):
                        ans = ret2[row][col+1]
                     if(ans<ret2[row+1][col-1]):
                        ans = ret2[row+1][col-1]
                     if(ans<ret2[row+1][col]):
                        ans = ret2[row+1][col]
                     if(ans<ret2[row+1][col+1]):
                        ans = ret2[row+1][col+1]
                     ret3[row][col] =ans
                     ans=-1000000
               
        cv2.imshow("I1",ret3)
        cv2.waitKey(0)
        #cv2.destoryAllWindows()
         
    def Show_Accuracy_and_Loss_VGG19(self):

        img = cv2.imread('training_curve_Vgg.png')
        cv2.imshow("I1",img)
        cv2.waitKey(0)

        
    def Show_Model_Structure_VGG19(self):
        device = torch.device("cuda")
        vgg19_bn = models.vgg19_bn(num_classes=10)
        vgg19_bn.to(device)
        summary(vgg19_bn, (3, 32, 32))
        cv2.waitKey(0)
        
#    def Predict_VGG19(self):

#        class_names=['0','1','2','3','4','5','6','7','8','9']

#        
#        if getattr(self,'pix',None) is None:
#            QMessageBox.warning(self, 'Warning','Please load image first!')
#            return
#        size = self.pix.size()
#        h = size.width()
#        w = size.height()
        
#        channels_count = 3
#        pixmap = self.pix
#        image = pixmap.toImage()
#        s = image.bits().asstring(w * h * channels_count)
#        arr = np.fromstring(s, dtype=np.uint8).reshape((h, w, channels_count))
#        img = transforms.ToTensor()(arr)
        
#        img = img.unsqueeze(0)
#        img = img.to(self.device)
#        with torch.no_grad():
                
#                pred = self.model_Vgg(img)
#                pred = torch.softmax(pred,dim=1)
#                pred_label_idx = pred.argmax(dim=1).item()
                
#        self.result_label.setText("Predicted Class ={}".format(class_names[pred_label_idx]))
#        pred =pred.squeeze().cpu().numpy()

#        plt.bar(range(10), pred)
#        plt.xlabel("Class")
#        plt.ylabel("Probability")
#        plt.title("Probability Distribution")
#        plt.show()
#        cv2.waitKey(0)
    def Predict_VGG19(self):

        class_names=['0','1','2','3','4','5','6','7','8','9']

        
        if getattr(self,'pix',None) is None:
            QMessageBox.warning(self, 'Warning','Please load image first!')
            return
        size = self.pix.size()
        h = size.width()
        w = size.height()
        
        #channels_count = 4
        #pixmap = self.pix

        #self.pix.pixmap().save('demo.png','PNG')

        
        #result = pixmap.copy()  # 複製整個 pixmap
        #Q_image = QtGui.QPixmap.toImage(result)
        #grayscale = Q_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
        #s = image.bits().asstring(w * h * channels_count)
        #arr = np.fromstring(s, dtype=np.uint8).reshape((h, w, channels_count))
        # 假設你的 pixmap 變數為 pix
        pix = self.pix

        # 轉換 QPixmap 為 QImage
        image = pix.toImage()

        # 將 QImage 轉換為 NumPy 數組
        size = image.size()
        width = size.width()
        height = size.height()
        channels_count = 4
        s = image.bits().asstring(width * height * channels_count)
        arr = np.fromstring(s, dtype=np.uint8).reshape((height, width, channels_count))

        # 將 NumPy 數組轉換為 Pillow 的 Image 物件
        image_pil = Image.fromarray(arr)
        gray_image_pil = image_pil.convert('L')
        if self.model_res is not None:
            try:
                transform = v2.Compose([
                            v2.Resize((32, 32)),
                            v2.ToTensor(),
                            v2.Normalize((0.1307,), (0.3081,)),
                            v2.Lambda(lambda x: x.repeat(3, 1, 1)),  
                ])
                

                #image = Image.open('demo.png')
                transformed_image = transform(gray_image_pil)
                #transformed_image = transform(image)
                transformed_image = transformed_image.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model_Vgg(transformed_image)
                
                _, predicted = torch.max(output, 1)
                
                self.result_label.setText(f"Predicted Class: {class_names[predicted.item()]}")

                # Plot probability distribution using a histogram
                probabilities = torch.softmax(output, dim=1).cpu().numpy()
                plt.bar(class_names, probabilities[0])

                plt.xlabel("Class")
                plt.ylabel("Probability")
                plt.title("Probability Distribution")
                plt.show()
                
            except Exception as e:
                print(f"Error during inference: {e}")
        
    def paintEvent(self, event):
        pp = QPainter(self.pix)
        # 根据鼠标指针前后两个位置绘制直线
        pen = QPen(QColor(Qt.white))
        pen.setWidth(5)  
        pp.setPen(pen) # Set pen color to white


        pp.drawLine(self.lastPoint - QPoint(500, 390), self.endPoint - QPoint(500, 390))  # 转换坐标
        # 让前一个坐标值等于后一个坐标值，
        # 这样就能实现画出连续的线
        self.lastPoint = self.endPoint
        painter = QPainter(self)
        #绘制画布到窗口指定位置处
        painter.drawPixmap(500, 390, self.pix)
      
    def mousePressEvent(self, event):
    # 鼠标左键按下
        if event.button() == Qt.LeftButton:
            self.lastPoint = event.pos()
            self.endPoint = self.lastPoint

    def mouseMoveEvent(self, event):
    # 鼠标左键按下的同时移动鼠标
        if event.buttons() and Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()

    def mouseReleaseEvent(self, event):
    # 鼠标左键释放
        if event.button() == Qt.LeftButton:
            self.endPoint = event.pos()
            # 进行重新绘制
            self.update()
    def QPixmapToArray(self):
        ## Get the size of the current pixmap
        size = self.pix.size()
        h = size.width()
        w = size.height()


        ## Get the QImage Item and convert it to a byte string
        qimg = self.pix.toImage()
        b = qimg.bits()
        # sip.voidptr must know size to support python buffer interface
        b.setsize(h * w * 4)
        arr = np.frombuffer(b, np.uint8).reshape((h, w, 4))

        return arr 

    
    def Reset_VGG19(self):

        self.pix.fill(Qt.black)
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.update()
       

        

    def _getitem_(self,index1,index2):
        img_name = self.img_names_c[index1]
        img_path = os.path.join(self.data_dir_c, img_name)
        img = Image.open(img_path)
        label = self.label_dir_c
        
        img_name = self.img_names_d[index2]
        img_path = os.path.join(self.data_dir_d, img_name)
        img2 = Image.open(img_path)
        label2 = self.label_dir_d
        
        return img, label ,img2,label2
    def Show_Images(self):
        dog_inference_dir = "../Hw2_C14096073_林星佑_V1/inference_dataset/Dog"
        cat_inference_dir = "../Hw2_C14096073_林星佑_V1/inference_dataset/Cat"
        transform = v2.Compose([
                            v2.Resize((224, 224))
                ])
        # Randomly select one dog and one cat image
        dog_image_path = os.path.join(dog_inference_dir, np.random.choice(os.listdir(dog_inference_dir), 1)[0])
        cat_image_path = os.path.join(cat_inference_dir, np.random.choice(os.listdir(cat_inference_dir), 1)[0])

        # Create a 1x2 subplot layout
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # Display the dog image
        dog_pic = Image.open(dog_image_path)
        
        dog_pic = transform(dog_pic)

        ax[0].imshow(dog_pic)
        ax[0].set_title('Dog')
        ax[0].set_axis_off()

        # Display the cat image
        cat_pic = Image.open(cat_image_path)
        cat_pic = transform(cat_pic)
        ax[1].imshow(cat_pic)
        ax[1].set_title('Cat')
        ax[1].set_axis_off()

        plt.show()
        cv2.waitKey(0)
        
    def Show_Model_Structure(self):
        device = torch.device("cuda")
        resnet50_model = models.resnet50(pretrained=True)

        in_features = resnet50_model.fc.in_features
        resnet50_model.fc = nn.Linear(in_features, 1)
        resnet50_model.fc = nn.Sequential(resnet50_model.fc, nn.Sigmoid())
        resnet50_model.to(device)    
        summary(resnet50_model, (3, 224, 224))
        cv2.waitKey(0)
        
    def inference_res(self):    
        if hasattr(self, 'image_path') and self.model_res is not None:
            try:
                transform = v2.Compose([
                            v2.ToTensor(),
                            v2.Resize((224, 224))
                ])
                
                image = Image.open(self.image_path)
                transformed_image = transform(image)

                transformed_image = transformed_image.unsqueeze(0).to(self.device)

                with torch.no_grad():
                    output = self.model_res(transformed_image)
                
                x, predicted = torch.max(output, 1)

                if (x<0.5):
                    self.result_label_2.setText(f"Cat")
                else:
                    self.result_label_2.setText(f"Dog")

                # Plot probability distribution using a histogram
                
            except Exception as e:
                print(f"Error during inference: {e}")

    
    def Show_Comparison(self):
        img = cv2.imread('compare.png')
        cv2.imshow("I1",img)
        cv2.waitKey(0)

    def load_model_Vgg(self, model_path_Vgg):
        try:
            self.model_Vgg = models.vgg19_bn(num_classes=10)
            self.model_Vgg.load_state_dict(torch.load(model_path_Vgg, map_location=self.device))
            self.model_Vgg.to(self.device)
            self.model_Vgg.eval()
        except Exception as e:
            print(f"Error loading the model: {e}")
    def load_model_res(self, model_path_res):
        try:
            self.model_res = models.resnet50(pretrained=True)
            in_features = self.model_res.fc.in_features
            self.model_res.fc = nn.Linear(in_features, 1)
            self.model_res.fc = nn.Sequential(self.model_res.fc, nn.Sigmoid())           
            self.model_res.load_state_dict(torch.load(model_path_res, map_location=self.device))

            self.model_res.to(self.device) 
            self.model_res.eval()         
        except Exception as e:
            print(f"Error loading the model: {e}")
        
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
    model_path_Vgg = "best_model_Vgg.pth"  # 请替换为您的模型文件路径
    model_path_res = "best_model_res.pth"  # 请替换为您的模型文件路径    
    root_dir = "../dataset/inference_dataset"
    mainWindow = Gui(model_path_Vgg,model_path_res)
    mainWindow.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
