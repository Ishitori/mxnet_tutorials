import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn

class ReluOp(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = nd.maximum(x, nd.zeros_like(x))
        self.assign(out_data[0], req[0], y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = out_data[0]
        dy = out_grad[0]
        dy_positives = nd.maximum(dy, nd.zeros_like(dy))
        y_ones = y.__gt__(0)
        dx = dy_positives * y_ones
        self.assign(in_grad[0], req[0], mx.nd.array(dx))

@mx.operator.register("relu")
class ReluProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ReluProp, self).__init__(True)

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = data_shape
        return (data_shape,), (output_shape,), ()

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return ReluOp()  

class Activation(mx.gluon.HybridBlock):
    def __init__(self, act_type, **kwargs):
        assert act_type == 'relu'
        super(Activation, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.Custom(x, op_type='relu')

class Conv2D(mx.gluon.HybridBlock):

    outputs = {}

    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, layout='NCHW',
                 activation=None, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', in_channels=0, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.conv = nn.Conv2D(channels, kernel_size, strides=strides, padding=padding,
                             dilation=dilation, groups=groups, layout=layout,
                             activation=activation, use_bias=use_bias, weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, in_channels=in_channels)
        self.output = None

    def hybrid_forward(self, F, x):
        out = self.conv(x)
        out.attach_grad()
        Conv2D.outputs[self._prefix[:-1]] = out
        return out

