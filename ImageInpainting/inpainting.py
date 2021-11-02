# from .lns import patch_reordering
from .pra import patch_reordering
from .images import read_image_as_arrays, save_arrays_as_image
from numpy import array, full, where, mean, arange
from numpy import random as rd
from scipy import interpolate

CORRUPTED_PIXEL = -999

def mean_of_squared_differences(patch1: array, patch2: array):
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

def cubic_spline(signal: array, mask: array):
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

class CorruptedImage:
    '''
    Class for model images with missing data
    defined by a mask. Provides a function to
    do a patches-based inpainting
    '''
    def __init__(self, location: str, mask_location=None, rgb=False, corrupt_prob=4/5):
        channels = read_image_as_arrays(location, rgb=rgb)
        self.shape = channels[0].shape
        self.rgb = rgb

        self.mask = read_image_as_arrays(mask_location, dtype=bool)[0] if isinstance(mask_location, str) else +\
                        rd.choice(a=[False, True], size=self.shape, p=[corrupt_prob, 1 - corrupt_prob])

        if self.shape != self.mask.shape:
            raise ValueError(f'Images at <{location}> and <{mask_location}> must have same resolution')

        corrupted = full(self.shape, CORRUPTED_PIXEL)
        self.channels = tuple(where(self.mask, ch, corrupted) for ch in channels)

    def save(self, location: str):
        save_arrays_as_image(self.channels, location=location, rgb=self.rgb)

    def inpainting(self, K=10, sqrt_n=16, B=9, epsilon=10**2, H=cubic_spline, omega=mean_of_squared_differences):     
        # get the image dimensions
        im_shape = self.shape
        n = sqrt_n * sqrt_n

        # get the dimensions of the subimage formed by corners of the patches
        Np1, Np2 = im_shape[0] - sqrt_n + 1, im_shape[1] - sqrt_n + 1
        Np = Np1 * Np2

        # get the mask version of the patches vector
        X_mask = array([
                    [self.mask[i:(i+sqrt_n), j:(j+sqrt_n)].flatten('F') 
                        for j in range(Np2)]
                            for i in range(Np1)
                ]).reshape(Np, n, order='F')

        # get the indices of subimages matrices
        X_pos = array([ 
                    [array([
                        [(ii, jj) 
                            for jj in range(j, j + sqrt_n)] 
                                for ii in range(i, i + sqrt_n)
                    ]).reshape(n, 2, order='F') 
                        for j in range(Np2)]
                            for i in range(Np1)
                ]).reshape(Np, n, 2, order='F')

        channels = []
        # iter by all channels
        for image in self.channels:
            # get the patches in matrix form
            Xm = array([
                    [image[i:(i+sqrt_n), j:(j+sqrt_n)].flatten('F')
                        for j in range(Np2)]
                            for i in range(Np1)
                ])

            # get the patches vector
            X = Xm.reshape(Np, n, order='F')
            
            # obtain K diferents reconstructions of the given image
            Ys = []

            for _ in range(K):
                # get the matrices Pk and Pk^(-1)
                per, iper = patch_reordering(patches=Xm, B=B, epsilon=epsilon, omega=omega)

                # permute X by Pk
                Xp = X[per]
                Xp_mask = X_mask[per]

                # apply the H operator to all subimages
                HXp = array([H(signal, mask) for signal, mask in zip(Xp.transpose(), Xp_mask.transpose())]).transpose()

                # reorder the patches by Pk^(-1)
                ys = HXp[iper]

                # aux matrix to store both, sums of pixels and its amount in each pos
                Yk_sum = full(im_shape, .0)
                Yk_cnt = full(im_shape, .0)

                # average the pixels of each sumimage into complete image pixels
                for y, xpos in zip(ys, X_pos):
                    for pixel, pos in zip(y, xpos):
                        Yk_sum[pos[0]][pos[1]] += pixel
                        Yk_cnt[pos[0]][pos[1]] += 1

                Ys.append(Yk_sum/Yk_cnt)

            # average this K result images to obtain the final image
            channels.append((1 / K) * sum(Ys))

        self.channels = tuple(channels)