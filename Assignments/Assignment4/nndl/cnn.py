import numpy as np

from nndl.layers import *
from nndl.conv_layers import *
from cs231n.fast_layers import *
from nndl.layer_utils import *
from nndl.conv_layer_utils import *

import pdb

""" 
This code was originally written for CS 231n at Stanford University
(cs231n.stanford.edu).  It has been modified in various areas for use in the
ECE 239AS class at UCLA.  This includes the descriptions of what code to
implement as well as some slight potential changes in variable names to be
consistent with class nomenclature.  We thank Justin Johnson & Serena Yeung for
permission to use this code.  To see the original version, please visit
cs231n.stanford.edu.  
"""

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype


        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize the weights and biases of a three layer CNN. To initialize:
        #     - the biases should be initialized to zeros.
        #     - the weights should be initialized to a matrix with entries
        #         drawn from a Gaussian distribution with zero mean and 
        #         standard deviation given by weight_scale.
        # ================================================================ #
        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
        self.params['b1'] = np.zeros(num_filters)
        pad = (filter_size - 1) / 2
        h_out,w_out = ((H-filter_size+2*pad+1),(W-filter_size+2*pad+1))
        h_outp,w_outp = (int((h_out-2)/2 +1),int((w_out-2)/2 +1))
        self.params['W2'] = weight_scale*np.random.randn((num_filters*h_outp*w_outp),hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)


        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the three layer CNN.  Store the output
        #   scores as the variable "scores".
        # ================================================================ #
        a1,cache_h1 = conv_forward_fast(X, W1, b1, conv_param)
        r1,_ = relu_forward(a1)
        p1,cache_h1p = max_pool_forward_fast(r1, pool_param)
        a2,_ = affine_forward(p1,W2,b2)
        r2,_ = relu_forward(a2)
        a3,_ = affine_forward(r2,W3,b3)
        scores = a3

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        if y is None:
            return scores

        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the three layer CNN.  Store the grads
        #   in the grads dictionary, exactly as before (i.e., the gradient of 
        #   self.params[k] will be grads[k]).  Store the loss as "loss", and
        #   don't forget to add regularization on ALL weight matrices.
        # ================================================================ #
        loss, dout = softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(np.square(W3)))
        dr, dw, db = affine_backward(dout,(r2,W3,b3))
        grads['W3']= dw + self.reg*W3
        grads['b3'] = db
        da = relu_backward(dr,a2)
        loss += 0.5*self.reg*(np.sum(np.square(W2)))
        dp, dw, db = affine_backward(da,(p1,W2,b2))
        grads['W2']= dw + self.reg*W2
        grads['b2'] = db
        dr = max_pool_backward_fast(dp, cache_h1p)
        da = relu_backward(dr,a1)
        loss += 0.5*self.reg*(np.sum(np.square(W1)))
        dx, dw, db = conv_backward_fast(da,cache_h1)
        grads['W1']= dw + self.reg*W1
        grads['b1'] = db

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads

class ConvNet(object):
    """
    A M+N-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=[7],
               hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_batchnorm=False):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.use_batchnorm = use_batchnorm
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.filt_size = filter_size


        # ================================================================ #
        # YOUR CODE HERE:
        #   Initialize the weights and biases of a three layer CNN. To initialize:
        #     - the biases should be initialized to zeros.
        #     - the weights should be initialized to a matrix with entries
        #         drawn from a Gaussian distribution with zero mean and 
        #         standard deviation given by weight_scale.
        # ================================================================ #
        C,H,W = input_dim
        conv_filters = np.array(filter_size)
        self.num_conv_layers = len(filter_size)
        ch = C
        h_outp = H
        w_outp = W
        for i in range(self.num_conv_layers):
            j = i+1
            wi = 'W'+ str(j)
            bi = 'b'+ str(j)
            self.params[wi] = weight_scale * np.random.randn(num_filters[i],ch,filter_size[i],filter_size[i])
            self.params[bi] = np.zeros(num_filters[i])
            ch = num_filters[i]
            pad = (filter_size[i] - 1) / 2
            h_out,w_out = ((h_outp-filter_size[i]+2*pad+1),(w_outp-filter_size[i]+2*pad+1))
            h_outp,w_outp = (int((h_out-2)/2 +1),int((w_out-2)/2 +1))
        
        if use_batchnorm:
            for i in range(self.num_conv_layers):
                j = i+1
                gammai = 'gamma'+str(j)
                betai = 'beta'+str(j)
                self.params[gammai] = np.ones(num_filters[i])
                self.params[betai] = np.zeros(num_filters[i])
        
        self.inp = num_filters[self.num_conv_layers-1]*h_outp*w_outp
        layers = np.array(hidden_dim)
        self.num_layers = 1 + len(hidden_dim)
        layers = np.insert(layers,(self.num_layers-1),num_classes)
        layers = np.insert(layers,0,self.inp)
    
        for i in range(self.num_layers):
            j = i+self.num_conv_layers +1
            wi = 'W'+ str(j)
            bi = 'b'+ str(j)
            self.params[wi] = weight_scale * np.random.randn(layers[i], layers[i+1])
            self.params[bi] = np.zeros(layers[i+1])
        
        if use_batchnorm:
            for i in range(self.num_layers-1):
                j = i+self.num_conv_layers + 1
                gammai = 'gamma'+str(j)
                betai = 'beta'+str(j)
                self.params[gammai] = np.ones(layers[i+1])
                self.params[betai] = np.zeros(layers[i+1])


        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #
        
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_conv_layers+self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        mode = 'test' if y is None else 'train'
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode

        # pass conv_param to the forward pass for the convolutional layer
        conv_params = []
        conv_params = [{'stride': 1, 'pad': (self.filt_size[i] - 1) / 2} for i in np.arange(self.num_conv_layers)]

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None

        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the forward pass of the three layer CNN.  Store the output
        #   scores as the variable "scores".
        # ================================================================ #
        infer = {}
        inp = X 
        for i in range(self.num_conv_layers+self.num_layers):
            wi = self.params['W'+str(i+1)]
            bi = self.params['b'+str(i+1)]
            if i<=(self.num_conv_layers-1):
                infer['a'+str(i+1)],infer['convc'+str(i+1)] = conv_forward_fast(inp,wi,bi,conv_params[i])
                if self.use_batchnorm:
                    gammai = self.params['gamma'+str(i+1)]
                    betai = self.params['beta'+str(i+1)]
                    infer['a'+str(i+1)],infer['cache'+str(i+1)] = spatial_batchnorm_forward(infer['a'+str(i+1)],gammai,betai,self.bn_params[i])
                infer['h'+str(i+1)],_ = relu_forward(infer['a'+str(i+1)])
                infer['p'+str(i+1)],infer['poolc'+str(i+1)] = max_pool_forward_fast(infer['h'+str(i+1)], pool_param)
                inp = infer['p'+str(i+1)]
            elif i==(self.num_conv_layers+self.num_layers-1):
                infer['a'+str(i+1)],_ = affine_forward(inp,wi,bi)
                scores = infer['a'+str(i+1)]
            else:
                infer['a'+str(i+1)],_ = affine_forward(inp,wi,bi)
                if self.use_batchnorm:
                    gammai = self.params['gamma'+str(i+1)]
                    betai = self.params['beta'+str(i+1)]
                    infer['a'+str(i+1)],infer['cache'+str(i+1)] =  batchnorm_forward(infer['a'+str(i+1)],gammai,betai,self.bn_params[i])
                infer['h'+str(i+1)],_ = relu_forward(infer['a'+str(i+1)])
                inp = infer['h'+str(i+1)]

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        if y is None:
            return scores

        loss, grads = 0, {}
        # ================================================================ #
        # YOUR CODE HERE:
        #   Implement the backward pass of the three layer CNN.  Store the grads
        #   in the grads dictionary, exactly as before (i.e., the gradient of 
        #   self.params[k] will be grads[k]).  Store the loss as "loss", and
        #   don't forget to add regularization on ALL weight matrices.
        # ================================================================ #
        idx = np.arange(self.num_conv_layers+self.num_layers)
        idx = np.flip(idx,axis=0)
        loss, dsmx = softmax_loss(scores,y)
        up_grad = dsmx
        for i in idx:
            wi = self.params['W'+str(i+1)]
            bi = self.params['b'+str(i+1)]
            loss += 0.5*self.reg*(np.sum(np.square(wi)))
            if (i>0 and i<self.num_conv_layers):
                dh, dw, db = conv_backward_fast(up_grad,infer['convc'+str(i+1)])
                dp = max_pool_backward_fast(dh, infer['poolc'+str(i)])
                da = relu_backward(dp,infer['a'+str(i)])
                if self.use_batchnorm:
                    cache = infer['cache'+str(i)]
                    da,grads['gamma'+str(i)],grads['beta'+str(i)] = spatial_batchnorm_backward(da,cache)
                up_grad = da
            elif i==0:
                dx, dw, db = conv_backward_fast(up_grad,infer['convc'+str(i+1)])
            elif i==self.num_conv_layers:
                dh, dw, db = affine_backward(up_grad,(infer['p'+str(i)],wi,bi))
                dp = max_pool_backward_fast(dh, infer['poolc'+str(i)])
                da = relu_backward(dp,infer['a'+str(i)])
                if self.use_batchnorm:
                    cache = infer['cache'+str(i)]
                    da,grads['gamma'+str(i)],grads['beta'+str(i)] = spatial_batchnorm_backward(da,cache)
                up_grad = da
            else:
                dh, dw, db = affine_backward(up_grad,(infer['h'+str(i)],wi,bi))
                da = relu_backward(dh,infer['a'+str(i)])
                if self.use_batchnorm:
                    cache = infer['cache'+str(i)]
                    da,grads['gamma'+str(i)],grads['beta'+str(i)] = batchnorm_backward(da,cache)
                up_grad = da
            grads['W'+str(i+1)]= dw + self.reg*wi
            grads['b'+str(i+1)] = db

        # ================================================================ #
        # END YOUR CODE HERE
        # ================================================================ #

        return loss, grads

