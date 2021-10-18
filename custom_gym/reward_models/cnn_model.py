
import torch
import torch.nn as nn
from torch.autograd import Variable

class Unit(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Unit,self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels,kernel_size=3,out_channels=out_channels,stride=1,padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet,self).__init__()

        #Create 14 layers of the unit with max pooling in between
        self.unit1 = Unit(in_channels=3,out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.unit2 = Unit(in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.unit3 = Unit(in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.unit4 = Unit(in_channels=128, out_channels=64)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.unit5 = Unit(in_channels=64, out_channels=32)
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv = nn.Conv2d(in_channels=32,kernel_size=3,out_channels=num_classes,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax()
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(   self.unit1, self.pool1, 
                                    self.unit2, self.pool2,
                                    self.unit3, self.pool3,
                                    self.unit4, self.pool4,
                                    self.unit5, self.pool5,
                                    self.conv, self.relu, self.avgpool, self.flatten, self.softmax)

    def forward(self, input):
        output = self.net(input)
        # output = output.view(-1,128)
        # output = self.fc(output)
        return output

class CNN_Model():
    def __init__(self):
        # CUDA for PyTorch
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        torch.backends.cudnn.benchmark = True
        self.model = None

    def load_model(self, path):
        checkpoint = torch.load(path)
        model = SimpleNet()
        model.load_state_dict(checkpoint)
        model.to(self.device)
        self.model = model

    def predict(self, img):
        self.model.eval()

        image_tensor = torch.tensor(img, device=self.device, dtype=torch.float)

        # Add an extra batch dimension since pytorch treats all images as batches
        image_tensor = image_tensor.unsqueeze_(0)

        # Turn the input into a Variable
        inputs = Variable(image_tensor)

        # Predict the class of the image
        with torch.no_grad():
            output = self.model(inputs)

        return output.data.cpu().numpy()[0]

