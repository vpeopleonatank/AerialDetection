import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shgeo
from dotadevkit.misc.dota_utils import dota_classes
from dotadevkit.polyiou import polyiou
from pycocotools import mask


def get_img_name_dict(ann_path):
    with open(ann_path, "r") as f:
        ann = json.load(f)
        img_name_dict = {}
        for img in ann["images"]:
            img_name_dict[img["file_name"]] = img

    return img_name_dict


def convert_nms_parsed_result_to_coco(
    det_pattern, classnames, image_parent, dest_file, ret_type, ann_file=""
):
    """Convert nms parsed result to coco json format"""

    data_dict = {}
    data_dict["images"] = []
    data_dict["categories"] = []
    data_dict["annotations"] = []
    segm_json_results = []
    for idex, name in enumerate(classnames):
        single_cat = {"id": idex + 1, "name": name, "supercategory": name}
        data_dict["categories"].append(single_cat)

    inst_count = 1
    image_id = 1

    if ann_file != "":
        img_name_dict = get_img_name_dict(ann_file)

    namebox_dict = {}
    for classname in classnames:
        print("classname:", classname)
        detfile = det_pattern.format(classname)
        with open(detfile, "r") as f:
            lines = f.readlines()

        splitlines = [x.strip().split(" ") for x in lines]
        for splitline in splitlines:
            image_name = splitline[0]
            score = float(splitline[1])
            bbox = [int(float(x)) for x in splitline[2:]]
            if image_name not in namebox_dict:
                namebox_dict[image_name] = []
            obj = {}
            obj["score"] = score
            obj["rbbox"] = bbox
            obj["class"] = classname
            namebox_dict[image_name].append(obj)

    for image_name in namebox_dict:
        imagepath = os.path.join(image_parent, image_name + ".png")
        img = cv2.imread(imagepath)
        height, width, _ = img.shape

        single_image = {}
        single_image["file_name"] = image_name + ".png"
        if ann_file != "":
            single_image["id"] = img_name_dict[single_image["file_name"]]["id"]
        else:
            single_image["id"] = image_id
        single_image["width"] = width
        single_image["height"] = height
        data_dict["images"].append(single_image)
        for det in namebox_dict[image_name]:
            single_obj = {}
            poly = shgeo.Polygon(tuple(zip(det["rbbox"][0::2], det["rbbox"][1::2])))
            single_obj["area"] = poly.area
            single_obj["category_id"] = classnames.index(det["class"]) + 1
            single_obj["segmentation"] = []
            single_obj["segmentation"].append(det["rbbox"])
            single_obj["iscrowd"] = 0
            xmin, ymin, xmax, ymax = (
                min(det["rbbox"][0::2]),
                min(det["rbbox"][1::2]),
                max(det["rbbox"][0::2]),
                max(det["rbbox"][1::2]),
            )

            width, height = xmax - xmin, ymax - ymin
            single_obj["bbox"] = xmin, ymin, width, height
            single_obj["image_id"] = single_image["id"]
            single_obj["id"] = inst_count
            data_dict["annotations"].append(single_obj)
            single_obj["segmentation"] = mask.frPyObjects(
                single_obj["segmentation"], height, width
            )
            segm_json_results.append(single_obj)

            inst_count = inst_count + 1
        image_id = image_id + 1
    with open(dest_file, "w") as f_out:
        if ret_type == "coco_ann":
            json.dump(data_dict, f_out)
        if ret_type == "coco_result":
            import ipdb; ipdb.set_trace()
            json.dumps(segm_json_results, f_out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert nms parsed result to coco format"
    )
    parser.add_argument(
        "--det-pattern", type=str, default=r"parsed_result/Task1_results_nms/{:s}.txt"
    )
    parser.add_argument("--ann-file", type=str, default=r"")
    parser.add_argument("--dest-file", type=str, default=r"res_coco.json")
    parser.add_argument("--image-parent", type=str, default=r"image_parent/")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "ship",
        ],
    )
    parser.add_argument(
        "--type",
        type=str,
        default="coco_ann",
        help="Conversation type in coco_ann or coco_result",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    convert_nms_parsed_result_to_coco(
        args.det_pattern,
        args.classes,
        args.image_parent,
        args.dest_file,
        args.type,
        args.ann_file,
    )
