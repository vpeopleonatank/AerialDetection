#!/bin/bash
export CONFIG_PATH_0_5m='configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_shipdata.py'
export CKPT_PATH_0_5m='work_dirs/epoch_8_0_5m.pth'

export CONFIG_PATH_3m='configs/DOTA/faster_rcnn_RoITrans_r50_fpn_1x_shipdata.py'
export CKPT_PATH_3m='work_dirs/epoch_1_3m.pth'

python app.py
