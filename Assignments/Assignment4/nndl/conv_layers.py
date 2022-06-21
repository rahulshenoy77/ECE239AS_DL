import numpy as np
from nndl.layers import *
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

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    pad = conv_param['pad']
    stride = conv_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the forward pass of a convolutional neural network.
    #   Store the output as 'out'.
    #   Hint: to pad the array, you can use the function np.pad.
    # ================================================================ #
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    F = w.shape[0]
    HH = w.shape[2]
    WW = w.shape[3]
    Hn = np.array((1 + (H + 2 * pad - HH) / stride),dtype='int')
    Wn = np.array((1 + (W + 2 * pad - WW) / stride),dtype='int')
    out = np.zeros((N,F,Hn,Wn),dtype='float')
    xp = np.pad(x,((0,0),(0,0),(pad,pad),(pad,pad)),'constant')
    mod1 = np.remainder((H+2*pad-HH),stride)
    mod2 = np.remainder((W+2*pad-WW),stride)
    if (mod1==0 and mod2==0):
        for i in range(F):
            s1 = 0
            e1 = HH
            s2 = 0
            e2 = WW
            for j in range(Hn):
                for k in range(Wn):
                    out[:,i,j,k]= np.sum(np.multiply(xp[:,:,s1:e1,s2:e2],w[i]),axis=(1,2,3)) + b[i]
                    s2 = s2+stride
                    e2 = e2+stride
                s1 = s1+stride
                e1 = e1+stride
                s2=0
                e2=WW
    else:
        raise ValueError('Invalid stride' )
    
    

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None

    N, F, out_height, out_width = dout.shape
    x, w, b, conv_param = cache

    stride, pad = [conv_param['stride'], conv_param['pad']]
    xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    num_filts, _, f_height, f_width = w.shape

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the backward pass of a convolutional neural network.
    #   Calculate the gradients: dx, dw, and db.
    # ================================================================ #
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    db = np.sum(dout,axis=(0,2,3))
    dw = np.zeros(w.shape)
    dxpad = np.zeros(xpad.shape)
    
    for l in range(N):
        for i in range(num_filts):
            s1 = 0
            e1 = f_height
            s2 = 0
            e2 = f_width
            for j in range(out_height):
                for k in range(out_width):
                    dw[i] += np.multiply(xpad[l,:,s1:e1,s2:e2],dout[l,i,j,k])
                    dxpad[l,:,s1:e1,s2:e2]+= np.multiply(w[i],dout[l,i,j,k])
                    s2 = s2+stride
                    e2 = e2+stride
                s1 = s1+stride
                e1 = e1+stride
                s2=0
                e2=f_width
    
    dx = dxpad[:,:,pad:pad+H,pad:pad+W]
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling forward pass.
    # ================================================================ #
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    Hn = np.array((1 + (H - pool_height) / stride),dtype='int')
    Wn = np.array((1 + (W - pool_width) / stride),dtype='int')
    out = np.zeros((N,C,Hn,Wn))
    s1 = 0
    e1 = pool_height
    s2 = 0
    e2 = pool_width
    for i in range(Hn):
        for j in range(Wn):
            out[:,:,i,j] = np.amax(x[:,:,s1:e1,s2:e2],axis=(2,3))
            s2 = s2+stride
            e2 = e2+stride
        s1 = s1+stride
        e1 = e1+stride
        s2 = 0
        e2 = pool_width
            
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 
    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the max pooling backward pass.
    # ================================================================ #
    N,C,out_height,out_width = dout.shape
    dx = np.zeros(x.shape)
    s1 = 0
    e1 = pool_height
    s2 = 0
    e2 = pool_width
    for i in range(out_height):
        for j in range(out_width):
            xc = x[:,:,s1:e1,s2:e2]
            dx[:,:,s1:e1,s2:e2] += (dout[:,:,i,j])[:,:,np.newaxis,np.newaxis] * (xc == np.amax(xc, axis = (2,3))[:,:,np.newaxis,np.newaxis])
            s2 = s2+stride
            e2 = e2+stride
        s1 = s1+stride
        e1 = e1+stride
        s2 = 0
        e2 = pool_width

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm forward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)
    
    N,C,H,W = x.shape
    xs = x.transpose(0,2,3,1).reshape((-1,C))

    Ns, Ds = xs.shape
    running_mean = bn_param.get('running_mean', np.zeros(Ds, dtype=xs.dtype))
    running_var = bn_param.get('running_var', np.zeros(Ds, dtype=xs.dtype))

    out, cache = None, None
    if mode == 'train':
        sample_mean = np.mean(xs,axis=0)
        sample_var = np.mean(((xs-sample_mean)**2),axis=0)
        running_mean = momentum*running_mean + (1-momentum)*sample_mean
        running_var = momentum*running_var + (1-momentum)*sample_var
        xn = np.divide((xs-sample_mean),(np.sqrt(sample_var+eps)))
        out = gamma*xn + beta
        out = out.reshape(N,H,W,C).transpose(0,3,1,2)
        cache = (xs,xn,gamma,sample_mean,sample_var,eps)
    elif mode == 'test':
        xn = np.divide((xs-running_mean),(np.sqrt(running_var+eps)))
        out = gamma*xn + beta
        out = out.reshape(N,H,W,C).transpose(0,3,1,2)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var


    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the spatial batchnorm backward pass.
    #
    #   You may find it useful to use the batchnorm forward pass you 
    #   implemented in HW #4.
    # ================================================================ #
    xs,xn,gamma,sample_mean,sample_var,eps = cache
    N,C,H,W = dout.shape
    douts = dout.transpose(0,2,3,1).reshape((-1,C))
    Ns = douts.shape[0]
    dbeta = np.sum(douts,axis=0)
    dgamma = np.sum((np.multiply(douts,xn)),axis=0)
    dxn = np.multiply(douts,gamma)
    dvar = -0.5*np.divide(np.sum(np.multiply((xs-sample_mean),dxn),axis=0),((sample_var+eps)**1.5))
    dmu = -1*np.divide(np.sum(dxn,axis=0),np.sqrt(sample_var+eps))
    dx = np.divide(dxn,np.sqrt(sample_var+eps)) + (2/Ns)*np.multiply((xs-sample_mean),dvar) + dmu/Ns
    dx = dx.reshape(N,H,W,C).transpose(0,3,1,2)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ # 

    return dx, dgamma, dbeta