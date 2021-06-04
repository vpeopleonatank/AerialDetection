import os
from glob import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Create empty label for images which don\'t have ship')
    parser.add_argument('--img-folder', help='image folder', type=str,
            default=r'images')
    parser.add_argument('--txt-folder', help='txt folder', type=str,
            default=r'txt')
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    img_paths = glob(args.img_folder + '/*.png')
    txt_paths = glob(args.txt_folder + '/*.txt')
    print('images don\'t have labels')
    for img_path in img_paths:
        img_basename = os.path.basename(img_path)
        img_basename = os.path.splitext(img_basename)[0]
        txt_basenames = [os.path.splitext(os.path.basename(txt_path))[0] for txt_path in txt_paths]
        if img_basename not in txt_basenames:
            print(os.path.join(args.img_folder, f"{img_basename}.png"))
            with open(os.path.join(args.txt_folder, f"{img_basename}.txt"), 'w') as f:
                f.write('')


if __name__ == '__main__':
    main()

