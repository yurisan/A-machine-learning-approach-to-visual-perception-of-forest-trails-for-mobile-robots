from PIL import Image
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers


class ColumnNet(chainer.Chain):
    def __init__(self, train=True):
        super(ColumnNet, self).__init__(
            conv1=L.Convolution2D(3, 32, 4),
            conv2=L.Convolution2D(32, 32, 4),
            conv3=L.Convolution2D(32, 32, 4),
            conv4=L.Convolution2D(32, 32, 3),
            l1=L.Linear(512, 200),
            l2=L.Linear(200, 3),
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        h = F.max_pooling_2d(F.relu(self.conv3(h)), 2)
        h = F.max_pooling_2d(F.relu(self.conv4(h)), 2)
        return self.l2(self.l1(h))
    
model = ColumnNet()
serializers.load_npz('columnnet.model', model)

img = np.array(Image.open("frame2072.jpg").resize((101,101)))
img = np.array([img.transpose(2,0,1)], dtype=np.float32)

print(model(img))
