from ImageInpainting import *
from time import perf_counter
from datetime import timedelta

import os
from re import match
import traceback

def recover_image(instance_dir: str):
    try:
        print(f'\nReading image "{instance_dir}" ...')
        ZR, ZG, ZB = read_image(instance_dir)
        Z_mask = ZR.astype(bool)

        print(f'Processing image "{instance_dir}" ...')
        
        start = last = perf_counter()
        for i, (YR, YG, YB) in enumerate(image_inpainting(ZR, ZG, ZB, Z_mask)):
            save_image(YR, YG, YB, instance_dir[:-4] + f'_recovered{i}.png')
            print(f'Iter#{i} ---> Duration: {timedelta(seconds=int(perf_counter() - last))}')
            last = perf_counter()

        print('Duration:', timedelta(seconds=int(last - start)))

    except Exception as ex:
        print(f'Error with "{instance}": {ex}')
        traceback.print_tb(ex.__traceback__)

if __name__ == '__main__':
    # corrupt_image('./Images/Tests/drake_no.png', './Images/Tests/test2.png')
    # corrupt_image('./Images/Tests/drake_yes.png', './Images/Tests/test3.png')

    instance = './Images/Tests/test'
    
    for i in range(2, 4):
        recover_image(instance + f'{i}.png')
        