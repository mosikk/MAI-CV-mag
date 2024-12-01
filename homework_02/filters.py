import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for m in range(Wk):
                    img_i = i + k - Hk // 2
                    img_j = j + m - Wk // 2
                    if img_i < 0 or img_i >= Hi or img_j < 0 or img_j >= Wi:
                        continue

                    out[i, j] += image[img_i, img_j] * kernel[Hk - k - 1, Wk - m - 1]

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    
    padded = zero_pad(image, Hk // 2, Wk // 2)
    kernel = np.flip(kernel)

    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(padded[i:i + Hk, j:j + Wk] * kernel)

    return out
    

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    pass

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_fast(f, np.flip(g))


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_fast(f, np.flip(g) - g.mean())


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    g = g.astype(np.float64)
    f = f.astype(np.float64)

    Hf, Wf = f.shape
    Hg, Wg = g.shape

    out = np.zeros((Hf, Wf))

    img_pad = zero_pad(f, Hg // 2, Wg // 2)

    g_std = np.std(g)
    g_mean = np.mean(g)
    g_norm = (g - g_mean) / g_std
    g_sum_sq = np.sum(g ** 2)

    for i in range(Hf):
        for j in range(Wf):
            img_slice = img_pad[i:i + Hg, j:j + Wg]

            img_slice_mean = np.mean(img_slice)
            img_slice_std = np.std(img_slice)
            img_slice_sum_sq = np.sum(img_slice ** 2)

            coeff = np.sqrt(g_sum_sq * img_slice_sum_sq)

            out[i, j] = np.sum(((img_slice - img_slice_mean) / img_slice_std) * g_norm) / coeff

    return out
