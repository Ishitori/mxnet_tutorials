import mxnet as mx
import numpy as np

def _get_grad(net, image, class_id=None, conv_layer_name=None, image_grad=False):

    if image_grad:
        image.attach_grad()
        Conv2D.capture_layer_name = None
    else:
        # Tell convviz.Conv2D which layer's output and gradient needs to be recorded
        Conv2D.capture_layer_name = conv_layer_name
    
    # Run the network
    with autograd.record(train_mode=False):
        out = net(image)
    
    # If user didn't provide a class id, we'll use the class that the network predicted
    if class_id == None:
        model_output = out.asnumpy()
        target_class = np.argmax(model_output)

    # Create a one-hot target with class_id and backprop with the created target
    one_hot_target = mx.nd.one_hot(mx.nd.array([target_class]), 1000)
    out.backward(one_hot_target, train_mode=False)

    if image_grad:
        return image.grad[0].asnumpy()
    else:
        # Return the recorded convolution output and gradient
        conv_out = Conv2D.conv_output
        return conv_out[0].asnumpy(), conv_out.grad[0].asnumpy()


def get_conv_out_grad(net, image, class_id=None, conv_layer_name=None):
    return _get_grad(net, image, class_id, conv_layer_name, image_grad=False)

def get_image_grad(net, image, class_id=None):
    return _get_grad(net, image, class_id, image_grad=True)

def grad_to_image(gradient):
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    gradient = gradient[..., ::-1]
    return gradient

