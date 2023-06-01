import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset_local_variance import SegDataset
import cv2
import numpy as np
#**********************
class Square(nn.Module):
    #====================
    def forward(self, x):
        return torch.square(x)

#***************************************************************
# Creating a CNN class
#=====================================
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object
    #================================
    def __init__(self):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=15)
        self.esp_x2 = nn.Sequential(
            Square(),
            nn.AvgPool2d(kernel_size=7,stride=1),)
        self.esp2_x = nn.Sequential(
            nn.AvgPool2d(kernel_size=7,stride=1),
            Square())
        self.relu = nn.ReLU()
    #================================
    # Progresses data across layers
    def forward(self, x):
        x = self.conv_layer1(x)
        esp2_x=self.esp2_x(x)
        esp_x2=self.esp_x2(x)
        return torch.sub(esp_x2,esp2_x)
        #return esp_x2
#=================================================
if __name__=="__main__":
    model=ConvNeuralNet()
    seg_dataset_test=SegDataset(is_train=None,root_dir="test_imgs")
    test_loader = torch.utils.data.DataLoader(dataset = seg_dataset_test,
                                           batch_size = 1,
                                           shuffle = True)
    for i,batch in enumerate(test_loader):
       output=model(batch)
    import matplotlib.pyplot as plt
    plt.figure()
    img_test=output.squeeze(0).permute(2,1,0).detach().numpy()*255
    img_test=np.abs(img_test)
    print(img_test)
    img_final=(img_test>0.07).astype(int)
    cv2.imwrite("output.png",img_final*255)
    

                                
