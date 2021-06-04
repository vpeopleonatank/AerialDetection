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
    for img_path in img_paths:
        print(os.path.basename(img_path))


if __name__ == '__main__':
    main()

