from ImageInpainting import CorruptedImage, read_image_as_mask
from time import perf_counter
from datetime import timedelta

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
    iterations = config["iterations"]
    images = config["images"]
    corrupt_prob = 0
    
    if "masks" in config:
        masks_dirs = config["masks"]
        if isinstance(masks_dirs, str):
            mask = read_image_as_mask(folder + masks_dirs)
            masks = [mask] * len(images)
        elif isinstance(masks_dirs, list):
            assert len(images) == len(masks_dirs), 'Bad json, different number of images and masks'
            masks = [read_image_as_mask(folder + mask_dir) for mask_dir in masks_dirs]
        else:
            assert False, 'Bad json, masks field must be str or list'
    else:   
        corrupt_prob = config["corrupt_prob"]
        masks = [None] * len(images)
    
    for image_dir, mask in zip(images, masks):
        try:
            image_dic = {}
            
            print(f'\nReading image "{image_dir}" ...')
            image, fmt = image_dir.rsplit('.', 1)
            cim = CorruptedImage(folder + image_dir, mask=mask, rgb=rgb, corrupt_prob=corrupt_prob)
            im = cv2.imread(folder + image_dir, flags=(cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE))

            print(f'Corrupting image "{image_dir}" ...')
            corrupted_dir = f'{image}_corrupted.{fmt}'
            cim.save(folder + corrupted_dir)
            corruped_pnsr = cv2.PSNR(im, cv2.imread(folder + corrupted_dir, flags=(cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)))

            print(f'Processing image "{image_dir}" ...')  
            iterations_dic = {}
            start = perf_counter()
            for iteration, kwargs in iterations.items():
                print(f'Iteration: {iteration} ---> Params: {kwargs}')
                
                last = perf_counter()
                cim.inpainting(**kwargs)
                duration = str(timedelta(seconds=int(perf_counter() - last)))
                
                iteration_dir = f'{image}_iteration_{iteration}.{fmt}'
                cim.save(folder + iteration_dir)
                psnr = cv2.PSNR(im, cv2.imread(folder + iteration_dir, flags=(cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)))
                
                print(f'Iteration: {iteration} ---> PSNR: {psnr} Duration: {duration}')
                iterations_dic[iteration] = { "psnr": psnr, "duration": duration }
            duration = str(timedelta(seconds=int(perf_counter() - start)))
            print('Duration:', duration)

            image_dic["corrupted"] = { "file": corrupted_dir, "psnr": corruped_pnsr}
            image_dic["iterations"] = iterations_dic
            image_dic["duration"] = duration
            
            results[image_dir] = image_dic

            with open(folder + 'results_' + json_dir, 'w') as js:
                json.dump(results, js, indent=4)
        except Exception as ex:
            print(f'Error with "{image}": {ex}')
            traceback.print_tb(ex.__traceback__)

        