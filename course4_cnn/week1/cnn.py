# Padding
def zero_pad(X, pad):
    return np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values=(0,0))

# Convolution Operation
def conv_single_step(a_slice_prev, W, b):
    # a_slice_prev is a slice of the activation input we want to apply filter W on
    return np.sum(a_slice_prev * W) + float(b)

def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev - f + (2 * pad)) / stride) + 1  
    n_W = int((n_W_prev - f + (2 * pad)) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                a_slice_prev = a_prev_pad[h * stride : h * stride + f, w * stride : w * stride + f, :]
                for c in range(n_C):
                    Z[i,h,w,c] = conv_single_step(a_slice_prev, W[...,c], b[...,c])
    
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def conv_backward(dZ, cache):
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']
    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    (m, n_H, n_W, n_C) = dZ.shape

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h
                    vert_end = vert_start + f
                    horiz_start = w
                    horiz_end = horiz_start + f

                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[...,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                    db[...,c] += dZ[i,h,w,c]

        dA_prev[i,...] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db


# Pooling
def pool_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters['f']
    stride = hparameters['stride']

    n_H = int((n_H_prev - f) / stride) + 1
    n_W = int((n_W_prev - f) / stride) + 1
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    a_prev_slice = A_prev[i, h*stride:h*stride+f, w*stride:w*stride+f, c]

                    A[i,h,w,c] = np.max(a_prev_slice) if mode=='max' else np.mean(a_prev_slice)

    cache = (A_prev, hparameters)

    return A, cache
    
