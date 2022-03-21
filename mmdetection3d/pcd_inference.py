# Copyright (c) OpenMMLab. All rights reserved.
from mmdet3d.core.points.lidar_points import LiDARPoints
from mmdet3d.apis import inference_detector, init_model

import os
import shutil
import time
import torch
import numpy as np

MODELS = {
    "second": {
        "config": "second/hv_second_secfpn_6x8_80e_kitti-3d-3class.py",
        "checkpoint": "hv_second_secfpn_6x8_80e_kitti-3d-3class_20200620_230238-9208083a.pth"
    },
    "dynamic_voxel": {
        "config": "dynamic_voxelization/dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class.py",
        "checkpoint": "dv_second_secfpn_2x8_cosine_80e_kitti-3d-3class_20200620_231010-6aa607d3.pth"
    },
    "parta2": {
        "config": "parta2/hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class.py",
        "checkpoint": "hv_PartA2_secfpn_2x8_cyclic_80e_kitti-3d-3class_20200620_230724-a2672098.pth"
    },
    "pointpillars": {
        "config": "pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py",
        "checkpoint": "hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth"
    }
}

# FILES = ["1002", "563", "2716", "2472", "2651", "1372", "1401", "482", "1262", "254"]
# FILES = ["278", "281", "254", "271", "326", "365", "366", "373", "392", "437", "505", "1757", "1771", "2204", "2217"]
FILES = ["278"]
def find_path(file_id, src_prefix="/wato/data/combined_cars/", dest_prefix="/wato/host/results/"):
        file_id = "0" * (5-len(file_id)) + file_id
        return (src_prefix + "pc" + file_id + ".npy", dest_prefix + "pc" + file_id)

def create_model(model_name):
    print(model_name)
    config = "configs/" + MODELS[model_name]["config"]
    checkpoint = "checkpoints/" + MODELS[model_name]["checkpoint"]
    return init_model(config, checkpoint)

def read_npy(file_path, to_lidar=True):
    pcd = np.load(file_path)
    zeros = np.zeros((pcd.shape[0], 1))
    pcd = np.hstack((pcd, zeros))
    pcd = np.float32(pcd)

    if (to_lidar):
        pcd = LiDARPoints(pcd, points_dim=pcd.shape[-1])
    return pcd


def load_test(model):

    file_paths = [find_path(x) for x in FILES[:4]]
    model_obj = create_model(model)

    pcds = [read_npy(x[0]) for x in file_paths]

    start = time.time()
    for i in range(0, 500):
        if (i % 100 == 0):
            print(torch.cuda.memory_allocated())
        for pcd in pcds:
            _result, _data = inference_detector(model_obj, pcd)
            
    end = time.time()
    print(model + ": " + str(end - start) + " for " + str(500 * len(file_paths)) + "files")


def main():
    file_paths = [find_path(x) for x in FILES]
    model_objs = [(x, create_model(x)) for x in MODELS.keys()]

    for file in file_paths:

        pcd = read_npy(file[0])

        if (os.path.isdir(file[1])):
            shutil.rmtree(file[1])
        os.mkdir(file[1])
        for model in model_objs:
            result, data = inference_detector(model[1], pcd) 

            points = data['points'][0][0].cpu().numpy()

            if 'pts_bbox' in result[0].keys():
                pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
                pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
                pred_labels = result[0]['pts_bbox']['labels_3d'].numpy()
            else:
                pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
                pred_scores = result[0]['scores_3d'].numpy()
                pred_labels = result[0]['labels_3d'].numpy()
            
            # np.save(file[1] + "/" + model[0] + "_points.npy", points)
            # np.save(file[1] + "/" + model[0] + "_preds.npy", pred_bboxes)
            # np.save(file[1] + "/" + model[0] + "_scores.npy", pred_scores)
            # np.save(file[1] + "/" + model[0] + "_labels.npy", pred_labels)


if __name__ == '__main__':
    main()
    # load_test("pointpillars")
