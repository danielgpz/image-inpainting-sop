from numpy import mean, arange
from scipy import interpolate


CORRUPTED_PIXEL = -999

def mean_of_squared_differences(patch1, patch2):
    '''
    Distance measure between patch1 and patch2 
    to be the average of squared differences 
    between existing pixels that share the same
    location in both patches.
    '''
    squared_differences = (patch1 - patch2)**2
    sub_squared_differences = squared_differences[(patch1 != CORRUPTED_PIXEL) & (patch2 != CORRUPTED_PIXEL)]

    if sub_squared_differences.size == 0:
        return -1

    return mean(sub_squared_differences)

def cubic_spline(signal, mask):
    '''
    Interpolation of signal using a mask to
    define indices to fill. Using cubic spline
    interpolation.
    '''
    x = arange(signal.size)
    masked_x = x[mask]

    if masked_x.size == x.size:
        return signal

    masked_signal = signal[mask]
    bspline = interpolate.splrep(masked_x, masked_signal)
    return interpolate.splev(x, bspline)