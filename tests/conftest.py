import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def data_dir(request):
    path = Path(request.module.__file__)
    data_path: Path = path.parent.joinpath('data')
    assert data_path.is_dir()
    return data_path


@pytest.fixture
def annot2():
    return np.asarray([
        [1002.9400, 930.4140],
        [978.2570, 1037.4400],
        [988.7030, 1065.7500],
        [1000.1400, 1061.8500],
        [1031.2800, 1075.4700],
        [974.5680, 1057.1000],
        [948.0630, 1032.6900],
        [938.8610, 1027.8300],
        [997.9838, 1250.9150],
        [998.1390, 1462.8400],
        [968.4760, 1631.6900],
        [964.9763, 1231.5650],
        [970.1040, 1441.6800],
        [941.3260, 1610.1600],
        [981.4800, 1241.2400],
        [975.5872, 1146.3147],
        [989.5510, 1002.4900]
    ])


@pytest.fixture
def annot3():
    return np.asarray([
        [-52.7130, -293.0790, 3626.7700],
        [-112.0150, -33.6677, 3611.9800],
        [-83.7290, 33.3961, 3483.2800],
        [-51.8461, 22.0815, 3161.5600],
        [12.8501, 47.0900, 2928.0000],
        [-125.9280, 14.3350, 3761.7800],
        [-208.1590, -50.8289, 4067.7900],
        [-246.9040, -67.7888, 4307.7300],
        [-61.7498, 465.9330, 3522.3050],
        [-63.6133, 985.6500, 3586.4500],
        [-135.3350, 1397.2600, 3604.8000],
        [-148.3182, 457.0890, 3756.5750],
        [-136.8890, 978.8790, 3754.9000],
        [-207.3250, 1389.9700, 3724.1300],
        [-105.0340, 461.5110, 3639.4400],
        [-118.6883, 231.0132, 3624.2306],
        [-84.9271, -118.1960, 3618.3000]
    ])


@pytest.fixture
def skeleton():
    from posekit.skeleton import skeleton_registry
    return skeleton_registry['mpi3d_17j']


@pytest.fixture
def coco_keypoints(data_dir):
    with data_dir.joinpath('coco_annots_350070.json').open('r') as f:
        annots = json.load(f)
    return torch.as_tensor(annots[5]['keypoints'], dtype=torch.float32).view(17, 3)


@pytest.fixture
def coco_image(data_dir):
    return Image.open(data_dir.joinpath('coco_image_350070.jpg'))
