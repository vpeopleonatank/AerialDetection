import os
from demo_large_image import DetectorModel
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import numpy as np
from typing import List
import uvicorn
import cv2
import datetime
from pycocotools import mask, _mask
import json


def create_image_info(
    image_id,
    file_name,
    image_size,
    date_captured=datetime.datetime.utcnow().isoformat(" "),
    license_id=1,
    coco_url="",
    flickr_url="",
):

    image_info = {
        "id": image_id,
        "file_name": file_name,
        "width": image_size[0],
        "height": image_size[1],
        "date_captured": date_captured,
        "license": license_id,
        "coco_url": coco_url,
        "flickr_url": flickr_url,
    }

    return image_info


def create_annotation_info(annotation_id, image_id, category_info, det, encoded_mask):

    # ptns = list([int(d) for d in det[:8]])
    ptns = det[:8].astype(int).tolist()

    area = mask.area(encoded_mask)
    if area < 1:
        return None

    bounding_box = mask.toBbox(encoded_mask).astype(int)

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": [ptns],
        "score": det[-1],
    }

    return annotation_info


def create_category_info(supercategory, id, name):
    category_info = {"supercategory": supercategory, "id": id, "name": name}

    return category_info


meta_info = {
    "year": 2021,
    "version": "1.0",
    "description": "Ship detection",
    "contributor": "",
    "url": "via",
    "date_created": "",
}

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ("ship",)


CONFIG_PATH_0_5m='configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_shipdata.py'
CKPT_PATH_0_5m='work_dirs/epoch_8_0_5m.pth'

CONFIG_PATH_3m='configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_shipdata.py'
CKPT_PATH_3m='work_dirs/epoch_1_3m.pth'
model_0_5m = DetectorModel(
    config_file=CONFIG_PATH_0_5m, checkpoint_file=CKPT_PATH_0_5m
)

model_3m = DetectorModel(
    config_file=CONFIG_PATH_3m, checkpoint_file=CKPT_PATH_3m
)

# def load_image_into_numpy_array(data):
#     return np.array(Image.open(BytesIO(data)))


def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


@app.post("/detectship")
async def upload_file(files: List[UploadFile] = File(...), model_type: str = Form(...)):
    print("Arrived")
    print(model_type)
    res = {"info": meta_info, "images": [], "annotations": [], "categories": []}

    # with open('/root/tmp/data/results.json', 'r') as f:
    #     data = json.load(f)

    # return data
    if model_type not in ["0_5", "3"]:
        return { "error": "specify model_type: 0_5 or 3" } 
    #model: DetectorModel
    if model_type == "0_5":
        model = model_0_5m
    elif model_type == "3":
        model = model_3m

    try:
        for i, name in enumerate(CLASSES):
            res["categories"].append(create_category_info(name, i + 1, name))
        image_id = 1
        annotation_id = 1
        for file in files:
            img = load_image_into_numpy_array(await file.read())
            height, width, _ = img.shape
            # detections = model.inference_single(img, (1024, 1024), (3072, 3072))
            detections = model.inference_single(img, (1024, 1024), (2048, 2048))
            res["images"].append(
                create_image_info(image_id, file.filename, (width, height))
            )
            if len(detections) != 0:
                for i, _ in enumerate(CLASSES):
                    dets = detections[i]
                    if dets.shape[0] == 0:
                        continue
                    # with open('/root/tmp/demo/dets.npy', 'wb') as f:
                    #     np.save(f, dets)
                    ptns = [det[:8] for det in dets]
                    # Return Run-length encoding of binary masks
                    masks = mask.frPyObjects(ptns, height, width)

                    for j, det in enumerate(dets):
                        if det[-1] < 0.3:
                            continue
                        ann = create_annotation_info(
                            annotation_id, image_id, res["categories"][0], det, masks[j]
                        )
                        if ann is None:
                            continue
                        res["annotations"].append(ann)
                        annotation_id += 1

            image_id += 1

        # with open('/root/tmp/data/results.json', 'w') as f:
        #     json.dump(res, f)

    except Exception as e:
        # import ipdb; ipdb.set_trace()
        print(e)
    # str_res = json.dumps(res)
    return res


if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=80, host="0.0.0.0")
