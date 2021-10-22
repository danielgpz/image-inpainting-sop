# from .lns import patch_reordering
from .pra import patch_reordering

from PIL import Image

from numpy import asarray, array
from numpy import random as rd
from numpy import fft

from scipy import interpolate


def read_image(location: str):
    '''
    Return the numpy array version
    of the image located at <location>
    on the disk
    '''

    img = Image.open(location).convert('RGB')
    R, G, B = (asarray(ch, dtype=int) for ch in img.split())

    return R, G, B

def save_image(R: array, G: array, B: array, location: str):
    '''
    Save a numpy array as an image at 
    <location> on the disk
    '''

    img = Image.merge('RGB', tuple(Image.fromarray(ch.astype(dtype='uint8')).convert('L') for ch in [R, G, B]))
    img.save(location)

def corrupt_image(load_location: str, save_location: str, prob=4/5):
    '''
    Take the image at <load_location> on disk
    and insert corrupt pixels randomly with
    probability <prob> and save the result
    at <save_location> on disk
    '''

    R, G, B = read_image(load_location)
    
    for rrow, grow, brow in zip(R, G, B):
        for i, _ in enumerate(rrow):
            if rd.choice([True, False], p=[prob, 1 - prob]):
                rrow[i] = grow[i] = brow[i] = 0
    
    save_image(R, G, B, save_location)

def distance_measure(patch1: array, patch2: array):
    ssum, scnt = 0, 0

    for val1, val2 in zip(patch1, patch2):
        if val1 >= 0 and val2 >= 0:
            diff = val1 - val2
            ssum += diff * diff
            scnt += 1

    return ssum / scnt if scnt else -1

def operator_h(row: array, row_mask: array):
    n = len(row)
    x = [i for i, bit in enumerate(row_mask) if bit]

    return interpolate.splev(range(n), interpolate.splrep(x, row.take(x)))

def image_process(image: array, image_mask: array, K=10, sqrt_n=16, B=9, e=10**2):
    # get the image dimensions
    N1, N2 = image.shape

    assert (N1, N2) == image_mask.shape, 'The 1st and 2nd image dimension must match with the dimension of image_mask'

    # get the dimensions of the subimage formed by corners of the patches
    N1p, N2p = N1 - sqrt_n + 1, N2 - sqrt_n + 1

    # saving for each parche from which pixel they come from
    X_pos = [[(p, q) for q in range(j, j + sqrt_n) for p in range(i, i + sqrt_n)]
                for j in range(N2p) for i in range(N1p)]

    # get the column stacked version of the image patches, the patches are in column stacked verion too
    X = [array([image[i][j] for  i, j in row]) for row in X_pos]
    X_mask = [array([image_mask[i][j] for  i, j in row]) for row in X_pos]
    
    # obtain K diferents reconstructions of the given image
    Ys = []

    for _ in range(K):
        # get the matrices Pk and Pk^(-1)
        # per, iper = patch_reordering(patches=X, w=distance_measure, duration=10.0)
        per, iper = patch_reordering(shapes=(N1p, N2p), patches=X, w=distance_measure, e=e, B=B)

        # permute X by Pk
        Xp = array(per(X))
        Xp_mask = array(per(X_mask))

        # apply the H operator to all subimages
        HXp = array([operator_h(xrow, mrow) for xrow, mrow in zip(Xp.T, Xp_mask.T)]).T

        # reorder the patches by Pk^(-1)
        _X = array(iper(HXp))

        Yk = [[[] for __ in range(N2)] for _ in range(N1)]

        # average the pixels of each sumimage into complete image pixels
        for patch, row in zip(_X, X_pos):
            for pixel, (i, j) in zip(patch, row):
                Yk[i][j].append(pixel)

        Ys.append(array([array([sum(elem) / len(elem) for elem in row]) for row in Yk]))

    # average this K result images to obtain the final image
    Y = (1 / K) * sum(Ys)

    return Y

def image_inpainting(R: array, G: array, B: array, image_mask: array):
    '''
    Return a new image recovering
    the missing pixels on <image>
    '''

    img = [R, G, B]

    Z = [array([array([pixel if bit else -1 for pixel, bit in zip(irow, mrow)]) 
                    for irow, mrow in zip(ch, image_mask)]) for ch in img]


    Y = [image_process(ch, image_mask, sqrt_n=3, B=3, e=10.0**2) for ch in Z]
    yield Y

    Y = [image_process(ch, image_mask, sqrt_n=3, B=3, e=10.0**8) for ch in Y]
    yield Y

    Y = [image_process(ch, image_mask, sqrt_n=3, B=3, e=10.0**8) for ch in Y]
    yield Y