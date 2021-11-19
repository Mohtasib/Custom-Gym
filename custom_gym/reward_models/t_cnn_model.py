
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
        self.unit1_common = Unit(in_channels=3,out_channels=64)
        self.pool1_common = nn.MaxPool2d(kernel_size=2)

        self.unit2_common = Unit(in_channels=64, out_channels=128)
        self.pool2_common = nn.MaxPool2d(kernel_size=2)

        self.unit3_common = Unit(in_channels=128, out_channels=256)
        self.pool3_common = nn.MaxPool2d(kernel_size=2)

        self.unit4_common = Unit(in_channels=256, out_channels=256)
        self.pool4_common = nn.MaxPool2d(kernel_size=2)



        self.unit1_class = Unit(in_channels=256, out_channels=128)
        self.pool1_class = nn.MaxPool2d(kernel_size=2)

        self.unit1_timing = Unit(in_channels=256, out_channels=128)
        self.pool1_timing = nn.MaxPool2d(kernel_size=2)



        self.unit2_class = Unit(in_channels=256, out_channels=64)
        self.pool2_class = nn.MaxPool2d(kernel_size=2)

        self.unit2_timing = Unit(in_channels=256, out_channels=64)
        self.pool2_timing = nn.MaxPool2d(kernel_size=2)



        self.conv_class = nn.Conv2d(in_channels=128,kernel_size=3,out_channels=2,stride=1,padding=1)
        self.relu_class = nn.ReLU()
        self.avgpool_class = nn.AdaptiveAvgPool2d(1)

        self.fc_class = nn.Linear(in_features=4, out_features=num_classes, bias=True)
        self.flatten_class = nn.Flatten()
        self.softmax_class = nn.Softmax()



        self.conv_timing = nn.Conv2d(in_channels=128,kernel_size=3,out_channels=2,stride=1,padding=1)
        self.relu_timing = nn.ReLU()
        self.avgpool_timing = nn.AdaptiveAvgPool2d(1)

        self.fc_timing = nn.Linear(in_features=4, out_features=1, bias=False)
        self.flatten_timing = nn.Flatten()


    def forward(self, x):
        x = self.pool1_common(self.unit1_common(x))
        x = self.pool2_common(self.unit2_common(x))
        x = self.pool3_common(self.unit3_common(x))
        x = self.pool4_common(self.unit4_common(x))

        C1 = self.pool1_class(self.unit1_class(x))
        T1 = self.pool1_timing(self.unit1_timing(x))

        C2 = torch.cat((C1, T1), 1)
        T2 = torch.cat((T1, C1), 1)

        C3 = self.pool2_class(self.unit2_class(C2))
        T3 = self.pool2_timing(self.unit2_timing(T2))
        
        C4 = torch.cat((C3, T3), 1)
        T4 = torch.cat((T3, C3), 1)

        C5 = self.avgpool_class(self.relu_class(self.conv_class(C4)))
        T5 = self.avgpool_timing(self.relu_timing(self.conv_timing(T4)))

        C6 = torch.cat((C5, T5), 2)
        T6 = torch.cat((T5, C5), 2)

        class_output = self.softmax_class(self.fc_class(self.flatten_class(C6)))
        timing_output = self.fc_timing(self.flatten_timing(T6))

        return class_output, timing_output

class T_CNN_Model():
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
            output, _ = self.model(inputs)

        return output.data.cpu().numpy()[0]

