import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
from mxnet import autograd

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

class Relu(mx.gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Relu, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.Custom(x, op_type='relu')

