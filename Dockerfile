### ORIGINAL DOCKERFILE (see mmdetection3d repo /docker)

ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# RUN pip install mmdet==2.17.0
# RUN pip install mmsegmentation==0.18.0

# Install MMDetection3D
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection3d.git /mmdetection3d_original
WORKDIR /mmdetection3d_original
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .


### CUSTOM WATO CHANGES

RUN apt-get update && apt-get install wget nano
RUN pip install mmdet==2.21.0 mmcv==1.3.18 mmsegmentation==0.21.1 mmcv-full==1.4.5 numpy

RUN pip install --user git+https://github.com/DanielPollithy/pypcd.git

WORKDIR /wato
RUN rm -rf /mmdetection3d_original
WORKDIR /wato/mmdetection3d
RUN mkdir checkpoints
RUN wget -P checkpoints/ https://download.openmmlab.com/mmdetection3d/v0.1.0_models/second/hv_second_secfpn_6x8_80e_kitti-3d-3class/hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth \
        https://download.openmmlab.com/mmdetection3d/v0.1.0_models/dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth \
        https://download.openmmlab.com/mmdetection3d/v0.1.0_models/parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth \
        https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth
