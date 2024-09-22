# Homework 1 

[Assignment Requirements Slide](https://github.com/hsylin/OpenCVDL/raw/main/HW1/OpenCv_Hw_1_Q_20231024_V1B4.pptx)

## Environment

- OS: Windows 10
- Python Version: 3.8

## Setup Instructions

1. Clone the repository:
   ```bash
   $ git clone https://github.com/hsylin/OpenCVDL.git
   ```
2. Install the required dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
3. Train the VGG19 model:
   ```bash
   $ python model.py
   ``` 
3. Run the application:
   ```bash
   $ python hw1.py
   ```

## Application Features

Once the application (`hw1.py`) is running, the UI is divided into five sections: **Image Processing**, **Image Smoothing**, **Edge Detection**, **Transforms**, and **VGG19**.

### 1. Image Processing
Offers three features:
- **Color Separation**: Extract and display the three BGR channels of "rgb.jpg" as separate grayscale images.
- **Color Transformation**: Convert "rgb.jpg" to grayscale (I1) using the OpenCV function cv2.cvtColor(), and then create an averaged image (I2) by merging the separated BGR channels.
- **Color Extraction**: Transform "rgb.jpg" to HSV format, extract a Yellow-Green mask, and create a modified image without yellow and green regions.

### 2. Image Smoothing
Provides three smoothing filters:
- **Gaussian Blur**: Adjust the Gaussian blur radius using a track bar.
- **Bilateral Filter**: Adjust the bilateral filter's radius using a track bar.
- **Median Filter**: Adjust the radius of the median filter using a track bar.

### 3. Edge Detection
Four options for edge detection:
- **Sobel X**: Detects vertical edges using the Sobel X operator.
- **Sobel Y**: Detects horizontal edges using the Sobel Y operator.
- **Combination and Threshold**: Combines Sobel X and Sobel Y images, applying a threshold to the result.
- **Gradient Angle**: Displays image areas where the gradient angle is between 120-180° and 210-330°.

### 4. Transforms
Performs image transformations:
- **Rotation, Scaling, and Translation**: Apply rotation, scaling, and translation to an image with adjustable parameters.

### 5. VGG19
Provides four features for classifying images into 10 different classes in CIFAR-10:
- **Show Augmented Images**: Displays nine augmented images from the `Q5_image/Q5_1` directory.
- **Show VGG19 Model Structure**: Displays the architecture of the VGG19 with Batch Normalization.
- **Show Training/Validation Accuracy and Loss**: Displays a graph of training and validation accuracy/loss.
- **Inference**: Load an image and use the trained model to classify it, displaying the predicted class label and probability distribution.

# Homework 2 

[Assignment Requirements Slide]
(https://github.com/hsylin/OpenCVDL/raw/main/HW2/OpenCv_Hw2_Q_20231205_V1B3.pptx)

## Environment

- OS: Windows 10
- Python Version: 3.8

## Setup Instructions

1. Clone the repository:
   ```bash
   $ git clone https://github.com/hsylin/OpenCVDL.git
   ```
2. Install the required dependencies:
   ```bash
   $ pip install -r requirements.txt
   ```
3. Train the VGG19 model:
   ```bash
   $ python VGG19.py
   ```
4. Train the ResNet50 model with random erasing:
   ```bash
   $ python ResNet_RE.py
   ```
5. Train the ResNet50 model without random erasing:
   ```bash
   $ python ResNet.py
   ```
6. Generate an accuracy comparison bar chart for the two ResNet50 models above:
   ```bash
   $ python ResNet_Compare.py
   ```
7. Run the application:
   ```bash
   $ python hw2.py
   ```

## Application Features

Once the application (`hw2.py`) is running, the UI is divided into five sections: **Hough Circle Transformation**, **Histogram Equalization**, **Morphology Operation**, **VGG19 MNIST Classifier**, and **ResNet50 Cat-Dog Classifier**.

### 1. Hough Circle Transformation
Offers two features:
- **Draw Contour**: Detect and draw the contours of circles in the loaded image.
- **Count Coins**: Detect the circle center points using the HoughCircles function and display the total number of coins present in the image.

### 2. Histogram Equalization
Provides one feature:
- **Apply Histogram Equalization**: Perform two types of histogram equalizations on the loaded image:
  1. OpenCV `cv2.equalizeHist()` method.
  2. A manual method using PDF and CDF.

### 3. Morphology Operation
Offers two features:
- **Closing Operation**: Manually implement the closing operation by applying dilation followed by erosion to fill small holes in the image.
- **Opening Operation**: Manually implement the opening operation by applying erosion followed by dilation to remove small objects or noise from the image.

### 4. VGG19 MNIST Classifier with Batch Normalization
Offers four features for handwritten digit recognition:
- **Show Model Structure**: Display the architecture of VGG19 with Batch Normalization.
- **Show Training/Validation Accuracy and Loss**: Show a plot of training and validation accuracy/loss.
- **Inference**: Draw a digit on the canvas and predict its class using the trained model. Also, display the probability distribution of the prediction.
- **Reset Canvas**: Clear the canvas for a new drawing.

### 5. ResNet50 Cat-Dog Classifier
Offers four features for classifying images of cats and dogs:
- **Show Random Images**: Display random images from the `inference_dataset/Cat` and `inference_dataset/Dog` directories.
- **Show ResNet50 Model Structure**: Display the architecture of the ResNet50 model.
- **Random Erasing**: Compare classification accuracy with and without random erasing as a data augmentation technique.
- **Inference**: Load an image, classify it using the trained model, and display the predicted class label.
