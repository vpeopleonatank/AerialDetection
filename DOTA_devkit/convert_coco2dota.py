import json
from pathlib import Path
import os
import argparse


def get_cat_mapping(cats):
    cats_dict = {}
    for cat in cats:
        cats_dict[int(cat["id"])] = cat["name"]

    return cats_dict

def get_ann_by_imgid(imgs, anns):
    ann_by_imgid_dict = {}
    for img in imgs:
        ann_by_imgid_dict[int(img['id'])] = []
        for ann in anns:
            if int(img['id']) == int(ann['image_id']):
                ann_by_imgid_dict[int(img['id'])].append(ann)

    return ann_by_imgid_dict


def convert_coco2dota(coco_ann, dota_out_path):
    os.makedirs(dota_out_path, exist_ok=True)
    with open(coco_ann, 'r') as f:
        coco_dict = json.load(f)
        idcat2name = get_cat_mapping(coco_dict["categories"])
        ann_by_imgid_dict = get_ann_by_imgid(coco_dict["images"], coco_dict["annotations"])
        for img in coco_dict["images"]:
            dota_anns = ""
            for anns in ann_by_imgid_dict[img["id"]]:
                # if int(ann["image_id"]) == int(img["id"]):
                if len(anns["segmentation"][0]) == 8:
                    for ptn in anns["segmentation"][0]:
                        dota_anns += str(ptn) + " "
                    dota_anns += idcat2name[int(anns["category_id"])] + " 0\n"
            with open(os.path.join(dota_out_path, Path(img["file_name"]).stem + '.txt'), 'w') as fw:
                fw.write(dota_anns)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco-path')
    parser.add_argument('--dota-save-path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    convert_coco2dota(args.coco_path, args.dota_save_path)


if __name__ == '__main__':
    main()
