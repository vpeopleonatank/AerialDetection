import os
from demo_large_image import DetectorModel
from fastapi import FastAPI, File, UploadFile
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


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):

    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
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
        "score": det[-1]
    } 

    return annotation_info

def create_category_info(supercategory, id, name):
    category_info = {
        "supercategory": supercategory,
        "id": id,
        "name": name
    }
    
    return category_info

meta_info = {
    "year": 2021,
    "version": "1.0",
    "description": "Ship detection",
    "contributor": "",
    "url": "via",
    "date_created": ""
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

CLASSES = ('ship',)
model = DetectorModel(config_file=os.getenv("CONFIG_PATH"), checkpoint_file=os.getenv("CKPT_PATH"))

# def load_image_into_numpy_array(data):
#     return np.array(Image.open(BytesIO(data)))
def load_image_into_numpy_array(data):
    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

@app.post('/detectship')
async def upload_file(files: List[UploadFile] = File(...)):
    print('Arrived')
    res = { "info": meta_info, "images": [], "annotations": [], "categories": []}

    try:
        for i, name in enumerate(CLASSES):
            res["categories"].append(create_category_info(name, i+1, name))
        image_id = 1
        annotation_id = 1
        for file in files:
            img = load_image_into_numpy_array(await file.read())
            width, height, _ = img.shape
            # detections = model.inference_single(img, (1024, 1024), (3072, 3072))
            detections = model.inference_single(img, (512, 512), (1024, 1024))
            res["images"].append(create_image_info(image_id, file.filename, (height, width)))
            for i, _ in enumerate(CLASSES):
                dets = detections[i]
                with open('/root/tmp/demo/dets.npy', 'rw') as f:
                    np.save(f, dets)
                ptns = [det[:8] for det in dets]
                masks = mask.frPyObjects(ptns, height, width)  # Return Run-length encoding of binary masks

                for j, det in enumerate(dets):
                    if det[-1] < 0.3:
                        continue
                    ann = create_annotation_info(annotation_id, image_id,
                        res["categories"][0],
                        det, masks[j])
                    if ann is None:
                        continue
                    res["annotations"].append(ann)
                    annotation_id += 1

            image_id += 1

    except Exception as e:
        import ipdb; ipdb.set_trace()
        print(e)
    # str_res = json.dumps(res)
    return res



if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=80, host="0.0.0.0")
