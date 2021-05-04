import os
# from demo_large_image import DetectorModel
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io, json
import base64
from typing import List
import uvicorn


app = FastAPI()


# model = DetectorModel(config_file=os.getenv("CONFIG_PATH"), checkpoint_file=os.getenv("CKPT_PATH"))

@app.post('/upload')
def upload_file(files: List[UploadFile] = File(...)):
    print('Arrived')

if __name__ == "__main__":
    # Run app with uvicorn with port and host specified. Host needed for docker port mapping
    uvicorn.run(app, port=8000, host="0.0.0.0")
