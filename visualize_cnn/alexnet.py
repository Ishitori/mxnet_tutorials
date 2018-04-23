import mxnet as mx
import mxnet.ndarray as nd
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import model_store

import numpy as np

class AlexNet(HybridBlock):
    def __init__(self, classes=1000, **kwargs):
        super(AlexNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(Conv2D(64, kernel_size=11, strides=4, padding=2))
                self.features.add(Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                
                self.features.add(Conv2D(192, kernel_size=5, padding=2))
                self.features.add(Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                
                self.features.add(Conv2D(384, kernel_size=3, padding=1))
                self.features.add(Activation('relu'))
                
                self.features.add(Conv2D(256, kernel_size=3, padding=1))
                self.features.add(Activation('relu'))
                
                self.features.add(Conv2D(256, kernel_size=3, padding=1))
                self.features.add(Activation('relu'))
                self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
                
                self.features.add(nn.Flatten())

                self.features.add(nn.Dense(4096))
                self.features.add(Activation('relu'))
                self.features.add(nn.Dropout(0.5))

                self.features.add(nn.Dense(4096))
                self.features.add(Activation('relu'))
                self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

def alexnet(pretrained=False, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    net = AlexNet(**kwargs)
    if pretrained:
        net.load_params(model_store.get_model_file('alexnet', root=root), ctx=ctx)
    return net

def preprocess(data):
    data = mx.image.imresize(data, 256, 256)
    data, _ = mx.image.center_crop(data, (224, 224))
    data = data.astype(np.float32)
    data = data/255
    data = mx.image.color_normalize(data,
                                    mean=mx.nd.array([0.485, 0.456, 0.406]),
                                    std=mx.nd.array([0.229, 0.224, 0.225]))
    data = mx.nd.transpose(data, (2,0,1))
    return data

