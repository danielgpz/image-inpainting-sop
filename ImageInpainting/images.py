from cv2 import imread, imwrite, IMREAD_GRAYSCALE, IMREAD_COLOR
from numpy import asarray, dsplit, dstack


def read_image_as_arrays(location: str, rgb=False, dtype=int):
    '''
    Return a tuple of numpy array version of
    the image located at <location> on the disk
    '''

    img = imread(location, IMREAD_COLOR if rgb else IMREAD_GRAYSCALE)

    if rgb:
        return (ch.resahpe(img.shape[:2], dtype=dtype) for ch in dsplit(img, 3))

    return (img.astype(dtype=dtype), )

def read_image_as_mask(location, true_value=0):
    return (imread(location, IMREAD_GRAYSCALE) == true_value)

def save_arrays_as_image(arrays: tuple, location: str, rgb=False):
    '''
    Save a tuple of numpy arrays as an image at 
    <location> on the disk
    '''

    if rgb:
        imwrite(location, dstack(arrays).clip(0, 255).astype('uint8'))
    else:
        imwrite(location, arrays[0].clip(0, 255).astype('uint8'))