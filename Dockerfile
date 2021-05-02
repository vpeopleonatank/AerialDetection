FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

WORKDIR /root/AerialDetection

ADD . /root/AerialDetection

RUN pip install cython ipykernel

RUN chmod +x ./compile.sh

RUN ./compile.sh

RUN pip install -r requirements.txt

RUN pip install -e .

RUN apt update
RUN apt install swig -y

WORKDIR /root/AerialDetection/DOTA_devkit
RUN swig -c++ -python polyiou.i
WORKDIR /root/AerialDetection
RUN python3 setup.py build_ext --inplace
