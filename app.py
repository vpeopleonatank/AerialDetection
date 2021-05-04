import os
from demo_large_image import DetectorModel
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import numpy as np
from typing import List
import uvicorn


app = FastAPI()


model = DetectorModel(config_file=os.getenv("CONFIG_PATH"), checkpoint_file=os.getenv("CKPT_PATH"))

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.post('/upload')
async def upload_file(files: List[UploadFile] = File(...)):
    print('Arrived')
    for file in files:
        img = load_image_into_numpy_array(await file.read())
        import ipdb; ipdb.set_trace()
        detections = model.inference_single(img, (1024, 1024), (3072, 3072))
        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=80, host="0.0.0.0")
