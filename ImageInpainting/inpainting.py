# from .lns import patch_reordering
from .images import read_image_as_arrays, save_arrays_as_image
from .operators import mean_of_squared_differences, cubic_spline, CORRUPTED_PIXEL
from .pra import patch_reordering
from numpy import array, full, where
from numpy import random as rd
from multiprocessing import Pool


def subimage_average_inpainting(
        im_shape: tuple, sim_shape: tuple, 
        X, X_mask, X_pos, B=9, epsilon=10**4, 
        H=cubic_spline, omega=mean_of_squared_differences
    ):
    '''
    Inpainting using the average of resulting 
    inpainting of each subimage. Auxiliar function
    only for multiprocessing reasons. 
    '''
    # get the matrices Pk and Pk^(-1)
    per, iper = patch_reordering(sim_shape, patches=X, B=B, epsilon=epsilon, omega=omega)

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

    return Yk_sum / Yk_cnt

class CorruptedImage:
    '''
    Class for model images with missing data
    defined by a mask. Provides a function to
    do a patches-based inpainting
    '''
    def __init__(self, location: str, mask=None, rgb=False, corrupt_prob=4/5):
        channels = read_image_as_arrays(location, rgb=rgb)
        self.shape = channels[0].shape
        self.rgb = rgb

        self.mask = rd.choice(
                        a=[False, True], 
                        size=self.shape, 
                        p=[corrupt_prob, 1 - corrupt_prob]
                    ) if mask is None else mask

        if self.mask.dtype != bool:
            raise ValueError(f'Mask must be an boolean ndarray')

        if self.shape != self.mask.shape:
            raise ValueError(f'Images at <{location}> and mask must have same resolution')

        corrupted = full(self.shape, CORRUPTED_PIXEL)
        self.channels = tuple(where(self.mask, ch, corrupted) for ch in channels)

    def save(self, location: str):
        save_arrays_as_image(self.channels, location=location, rgb=self.rgb)

    def inpainting(self, K=10, sqrt_n=16, B=9, epsilon=10**4, H=cubic_spline, omega=mean_of_squared_differences):     
        # get the image and patches dimensions
        N1, N2 = self.shape
        n = sqrt_n * sqrt_n

        # get the dimensions of the subimage formed by corners of the patches
        Np1, Np2 = N1 - sqrt_n + 1, N2 - sqrt_n + 1
        Np = Np1 * Np2

        # get the mask version of the patches vector
        X_mask = array([
                    [self.mask[i:(i + sqrt_n), j:(j + sqrt_n)].flatten('F') 
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
            # get the patches vector
            X = array([
                    [image[i:(i + sqrt_n), j:(j + sqrt_n)].flatten('F')
                        for j in range(Np2)]
                            for i in range(Np1)
                ]).reshape(Np, n, order='F')

            # obtain K diferents reconstructions of the given image
            Ys, args = [], ((N1, N2), (Np1, Np2), X, X_mask, X_pos, B, epsilon, H, omega)
            # using a multiprocessing pool
            with Pool(processes=3) as pool:
                for it in range(0, K, 3):
                    # run a subimage average inpainting each time
                    procs = [pool.apply_async(subimage_average_inpainting, args) for p in range(3) if it + p < K]
                    for proc in procs:
                        Ys.append(proc.get())

            # average this K result images to obtain the final image
            channels.append((1 / K) * sum(Ys))

        self.channels = tuple(channels)