import torch.nn as nn
import torch.nn.functional as F


VGG_types = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


class VGG(nn.Module):
    '''
    implementation of modular VGG architecture models -> NOT in use for final project.
    '''
    def __init__(self, num_classes, in_channels, architecture):
        '''
        constructor
        @param num_classes: (int) number of classes to train on.
        @param in_channels: (int) number of in channels.
        @param architecture: (str) which architecture the model will be -> key from VGG_types.
        '''
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_layers(VGG_types[architecture]) # build the layers

        self.fc = nn.Sequential( # fully connected layers.
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        '''
        pass input through the network.
        @param x: input image.
        @return: index of prediction.
        '''
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return F.softmax(x, dim=1)

    def create_layers(self, architecture):
        '''
        create the layers of the network according to the architecture provided.
        @param architecture: (str) key from VGG_types
        @return: (torch.nn.Sequential) list of layers in the network.
        '''
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU()]
                in_channels = layer
            elif layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)