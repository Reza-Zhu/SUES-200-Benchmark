import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from hardnet_model import HardNet
# import kornia.feature as KF
# from kornia.feature import laf_from_center_scale_ori as get_laf
from extract_patches.core import extract_patches


def extract_sift_keypoints_upright(img, n_feat=5000):
    sift = cv2.SIFT_create(
        # 2 * n_feat,
            contrastThreshold=-10000, edgeThreshold=-10000
    )
    keypoints = sift.detect(img, None)
    response = np.array([kp.response for kp in keypoints])
    respSort = np.argsort(response)[::-1]
    kpts = [cv2.KeyPoint(keypoints[i].pt[0], keypoints[i].pt[1], keypoints[i].size, 0) for i in respSort]
    kpts_unique = []
    for x in kpts:
        if x not in kpts_unique:
            kpts_unique.append(x)
    return kpts_unique[:n_feat]

def extract_descriptors(kpts, img, As, descnet,dev):
    # descnet = descnet.to(dev)
    # descnet.eval()
    patches = np.array(extract_patches((kpts, As),
                                       cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
                                       32, 12., 'cv2+A')).astype(np.float32)
    bs = 128
    desc = np.zeros((len(patches), 128))
    for i in range(0, len(patches), bs):
        data_a = torch.from_numpy(patches[i:min(i + bs, len(patches)), :, :]).unsqueeze(1).to(dev)
        with torch.no_grad():
            out_a = descnet(data_a)
            desc[i:i + bs] = out_a.cpu().detach().numpy()
    return desc

def detect_sift_HardNet(img, kpts, model, dev=torch.device('cuda')):
    hardnet = model
    As = torch.eye(2).view(1, 2, 2).expand(len(kpts), 2, 2).numpy()
    descs = extract_descriptors(kpts, img, As, hardnet,dev)
    return descs