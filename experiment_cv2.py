from time import perf_counter
from datetime import timedelta


import numpy as np
import sys, json, cv2, traceback

if __name__ == '__main__':

    try:
        json_dir = sys.argv[1]
    except:
        exit(0)

    with open(json_dir, 'r') as js:
        config = json.load(js)

    results = {}
    folder = config["folder"]
    rgb = config["rgb"]
    images = config["images"]
    corrupt_prob = 0
    
    if "masks" in config:
        masks_dirs = config["masks"]
        if isinstance(masks_dirs, str):
            mask = cv2.imread(folder + masks_dirs, cv2.IMREAD_GRAYSCALE)
            masks = [mask] * len(images)
        elif isinstance(masks_dirs, list):
            assert len(images) == len(masks_dirs), 'Bad json, different number of images and masks'
            masks = [cv2.imread(folder + mask_dir, cv2.IMREAD_GRAYSCALE) for mask_dir in masks_dirs]
        else:
            assert False, 'Bad json, masks field must be str or list'
    else:   
        assert False, 'Bad json, missing masks field'
    
    for image_dir, mask in zip(images, masks):
        try:
            image_dic = {}
            
            print(f'\nReading image "{image_dir}" ...')
            image, fmt = image_dir.rsplit('.', 1)
            im = cv2.imread(folder + image_dir, flags=(cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE))

            radius = 8
            print(f'Processing image "{image_dir}" ...')  
            iterations_dic = {}
            start = perf_counter()
            for itype, iflag in [('TELEA', cv2.INPAINT_TELEA), ('NS', cv2.INPAINT_NS)]:
                print(f'OpenCV inpainting {itype} with radius: {radius}')
                
                last = perf_counter()
                rim = cv2.inpaint(im, mask, radius, flags=iflag)
                duration = str(timedelta(seconds=int(perf_counter() - last)))
                
                iteration_dir = f'{image}_cv2_{itype}.{fmt}'
                cv2.imwrite(folder + iteration_dir, rim)
                psnr = cv2.PSNR(im, rim)
                
                print(f'-----------> PSNR: {psnr} Duration: {duration}')
                iterations_dic[itype] = { "psnr": psnr, "duration": duration }
            duration = str(timedelta(seconds=int(perf_counter() - start)))
            print('Duration:', duration)

            image_dic["iterations"] = iterations_dic
            image_dic["duration"] = duration
            
            results[image_dir] = image_dic

            with open(folder + 'cv2_results_' + json_dir, 'w') as js:
                json.dump(results, js, indent=4)
        except Exception as ex:
            print(f'Error with "{image}": {ex}')
            traceback.print_tb(ex.__traceback__)

        