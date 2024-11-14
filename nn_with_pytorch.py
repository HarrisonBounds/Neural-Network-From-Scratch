import torch.nn as nn

class NeuralNet(nn.module):
    def __init__(self, input_size, hl_size, output_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hl_size)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hl_size, output_size)
        self.softmax2 = nn.Softmax()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid1(x)
        x = self.linear2
        x = self.softmax2(x)
        return x
        
        
        