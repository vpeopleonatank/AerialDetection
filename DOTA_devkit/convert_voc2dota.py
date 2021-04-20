import cv2
import numpy as np
import math
import skimage.io
import xml.etree.ElementTree as ET
import glob
import os
import shutil

def read_label(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    filename = root.find('filename').text
    for boxes in root.iter('object'):

        cx = float(boxes.find("robndbox/cx").text)
        cy = float(boxes.find("robndbox/cy").text)
        w = float(boxes.find("robndbox/w").text)
        h = float(boxes.find("robndbox/h").text)
        angle = float(boxes.find("robndbox/angle").text)
        name = str(boxes.find("name").text)

        list_with_single_boxes = [cx, cy, w, h, angle, name]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes


def rotatePoint(xc,yc, xp,yp, theta):        
    xoff = xp-xc;
    yoff = yp-yc;

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    # pRes = (xc + pResx, yc + pResy)
    return xc+pResx,yc+pResy

def read_images(img_path, label_path):
    # img = cv2.imread(img_path) 
    # resized_img = cv2.resize(img, (512, 1024))
    # cv2.imshow('resized_img', resized_img)
    _, boxes = read_label(label_path)
    blank_img = np.zeros((600, 800, 3))
    box = boxes[0]
    cx, cy, w, h, angle = box[0], box[1], box[2], box[3], box[4]
    p0x,p0y = rotatePoint(cx,cy, cx - w/2, cy - h/2, -angle)
    p1x,p1y = rotatePoint(cx,cy, cx + w/2, cy - h/2, -angle)
    p2x,p2y = rotatePoint(cx,cy, cx + w/2, cy + h/2, -angle)
    p3x,p3y = rotatePoint(cx,cy, cx - w/2, cy + h/2, -angle)
    cv2.circle(blank_img, (int(p0x), int(p0y)), 3, (255,0,0))
    cv2.circle(blank_img, (int(p1x), int(p1y)), 3, (0,255,0))
    cv2.circle(blank_img, (int(p2x), int(p2y)), 3, (0,0,255))
    cv2.circle(blank_img, (int(p3x), int(p3y)), 3, (255,255,0))
    cv2.imshow('img', blank_img)
    

def convert_voc2dota(folder_path, folder_out_path):
    image_paths = glob.glob(f"{folder_path}/*.png")
    
    import ipdb; ipdb.set_trace()
    for image_path in image_paths:
        base_path = f'{os.path.dirname(image_path)}'
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = f"{base_path}/{file_name}.xml"
        _, boxes = read_label(label_path)
        with open(f'{folder_out_path}/{file_name}.txt', 'w') as f:
            for box in boxes:
                cx, cy, w, h, angle = box[0], box[1], box[2], box[3], box[4]
                p0x,p0y = rotatePoint(cx,cy, cx - w/2, cy - h/2, -angle)
                p1x,p1y = rotatePoint(cx,cy, cx + w/2, cy - h/2, -angle)
                p2x,p2y = rotatePoint(cx,cy, cx + w/2, cy + h/2, -angle)
                p3x,p3y = rotatePoint(cx,cy, cx - w/2, cy + h/2, -angle)
                name = box[5]
                if angle <= math.pi / 2:
                    f.write(f"{str(p3x)} {str(p3y)} {str(p0x)} {str(p0y)} {str(p1x)} {str(p1y)} {str(p2x)} {str(p2y)} {name} 0\n")
                else:
                    f.write(f"{str(p1x)} {str(p1y)} {str(p2x)} {str(p2y)} {str(p3x)} {str(p3y)} {str(p0x)} {str(p0y)} {name} 0\n")

def split_data_have_label(src_path, out_path):
    image_parent = os.path.join(src_path, 'images')
    label_parent = os.path.join(src_path, 'labelTxt')
    image_paths = glob.glob(f"{image_parent}/*.png")

    out_image_parent = f"{out_path}/images"
    out_label_parent = f"{out_path}/labelTxt"
    os.makedirs(out_image_parent, exist_ok=True)
    os.makedirs(out_label_parent, exist_ok=True)

    count = 0
    for image_path in image_paths:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = f"{label_parent}/{file_name}.txt"
        with open(label_path, "r") as f:
        
            if f.readline() != "":
                count += 1
                shutil.copy(image_path, f"{out_image_parent}/{file_name}.png")
                shutil.copy(label_path, f"{out_label_parent}/{file_name}.txt")

    print(f"{count} images have labels")

def train_val_split(src_path, out_path, val_ratio=0.1):
    """
    Split valid data based on the n-th first images
    """
    os.makedirs(out_path, exist_ok=True)
    
    image_parent = os.path.join(src_path, 'images')
    label_parent = os.path.join(src_path, 'labelTxt')
    image_paths = glob.glob(f"{image_parent}/*.png")

    out_train_parent = f"{out_path}/train"
    os.makedirs(out_train_parent, exist_ok=True)
    out_val_parent = f"{out_path}/val"
    os.makedirs(out_val_parent, exist_ok=True)

    out_image_train_parent = f"{out_train_parent}/images"
    out_label_train_parent = f"{out_train_parent}/labelTxt"
    os.makedirs(out_image_train_parent, exist_ok=True)
    os.makedirs(out_label_train_parent, exist_ok=True)

    out_image_val_parent = f"{out_val_parent}/images"
    out_label_val_parent = f"{out_val_parent}/labelTxt"
    os.makedirs(out_image_val_parent, exist_ok=True)
    os.makedirs(out_label_val_parent, exist_ok=True)

    val_num = int(val_ratio * len(image_paths))
    count = 0
    for image_path in image_paths:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = f"{label_parent}/{file_name}.txt"

        if count < val_num:
            shutil.copy(image_path, f"{out_image_val_parent}/{file_name}.png")
            shutil.copy(label_path, f"{out_label_val_parent}/{file_name}.txt")
            count += 1
        else:
            shutil.copy(image_path, f"{out_image_train_parent}/{file_name}.png")
            shutil.copy(label_path, f"{out_label_train_parent}/{file_name}.txt")
            count += 1

    print(f"train dataset: {len(image_paths) - val_num} images\nvalid dataset: {val_num} images")


def main():
    # img_path = '/mnt/Data/Project/ShipDetection/Data_Ship/AnnotatedData/20190308_060133_ssc10_u0001[DL].png'
    # label_path =  '/mnt/Data/Project/ShipDetection/Data_Ship/AnnotatedData/20190308_060133_ssc10_u0001[DL].xml' 
    # label_path = '/home/vpoat/Desktop/vis2.xml'
    # read_label(label_path)
    # read_images(img_path, label_path)
    folder_path = '/mnt/Data/Project/ShipDetection/Data_Ship/test_raw_data'
    out_folder = '/mnt/Data/Project/ShipDetection/Data_Ship/test_raw_data'
    convert_voc2dota(folder_path, out_folder)

    # split_data_have_label('/mnt/Data/Project/ShipDetection/Data_Ship/DOTA_splitted_1024/',
    #                 '/mnt/Data/Project/ShipDetection/Data_Ship/DOTA_truncated_splitted_1024/')

    # train_val_split('/mnt/Data/Project/ShipDetection/Data_Ship/DOTA_truncated_splitted_1024/',
    #                 '/mnt/Data/Project/ShipDetection/Data_Ship/DOTA_truncated_splitted_1024/')




if __name__ == '__main__':
    main()


