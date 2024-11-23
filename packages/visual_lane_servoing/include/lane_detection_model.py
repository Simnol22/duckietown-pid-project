
import torch
import torch.nn as nn

# preprocessing
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image

class LaneDetectionCNN(nn.Module):
    def __init__(self, input_shape=(3, 224, 224)):
        super(LaneDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flat size dynamically
        self._to_linear = None
        self._calculate_flat_size(input_shape)

        self.fc1 = nn.Linear(self._to_linear, 7*7*3) # 7x7 image with 3 channels
        self.fc2 = nn.Linear(7*7*3, 1)  # Single output neuron for regression

    def _calculate_flat_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self._forward_conv(x)
        self._to_linear = x.numel()

    def _forward_conv(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = torch.relu(self.conv4(x))
        x = self.maxpool3(x)
        x = torch.relu(self.conv5(x))
        x = self.maxpool4(x)
        x = self.dropout(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = x*0.5 #Scale output to be between -0.5 and 0.5
        return x
    
def apply_preprocessing(image):
    """
    Apply preprocessing transformations to the input image.

    Parameters:
    - image: PIL Image object.
    """
    
    image_array = np.array(image)
    channels = [image_array[:, :, i] for i in range(image_array.shape[2])]
    
    imghsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    
    img = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    mask_ground = np.ones(img.shape, dtype=np.uint8)  # Start with a mask of ones (white)
    mid_point = img.shape[0] // 2  # Integer division to get the middle row index

    # Set the top half of the image to 0 (black)
    mask_ground[:mid_point-30, :] = 0  # Mask the top half (rows 0 to mid_point-1)
    
    #gaussian filter
    sigma = 3.5
    img_gaussian_filter = cv2.GaussianBlur(img,(0,0), sigma)
    
    sobelx = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img_gaussian_filter,cv2.CV_64F,0,1)
    # Compute the magnitude of the gradients
    Gmag = np.sqrt(sobelx*sobelx + sobely*sobely)
    threshold = 35
    mask_mag = (Gmag > threshold)
        #4 Mask yellow and white

    white_lower_hsv = np.array([0,(0*255)/100,(60*255)/100]) # [0,0,50] - [230,100,255]
    white_upper_hsv = np.array([150,(40*255)/100,(100*255)/100])   # CHANGE ME

    yellow_lower_hsv = np.array([(30*179)/360, (30*255)/100, (30*255)/100])        # CHANGE ME
    yellow_upper_hsv = np.array([(90*179)/360, (110*255)/100, (100*255)/100])  # CHANGE ME
    mask_white = cv2.inRange(imghsv, white_lower_hsv, white_upper_hsv)
    mask_yellow = cv2.inRange(imghsv, yellow_lower_hsv, yellow_upper_hsv)


    final_mask = mask_ground * mask_mag  * (mask_white + mask_yellow)
    # Convert the NumPy array back to a PIL image
    for channel in channels:
        channel *= final_mask
    filtered_image = np.stack(channels, axis=-1)
    filtered_image = Image.fromarray(filtered_image)
    return filtered_image

def apply_preprocess(image):
    # takes in image from camera and outputs image ready for inference
    
    #image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float() / 255.0
    # need preprocessing
    transform = transforms.Compose([
        transforms.Lambda(apply_preprocessing),
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # Convert to tensor without resizing
    ])

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor