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

def extract_descriptors(kpts, img, As, descnet, dev=torch.device('cpu')):
    descnet = descnet.to(dev)
    descnet.eval()
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

def detect_sift_HardNet(img, nfeats = 5000, dev=torch.device('cpu')):
    hardnet = HardNet(True).to(dev).eval()
    # hardnet = KF.HardNet(True).to(dev).eval()
    # orinet = torch.jit.load('OriNetJIT.pt').to(dev).eval()
    # orinet.eval()
    kpts = extract_sift_keypoints_upright(img, nfeats)
    As = torch.eye(2).view(1, 2, 2).expand(len(kpts), 2, 2).numpy()
    # ori = estimate_orientation(kpts, img, As, orinet, dev)
    # kpts_new = [cv2.KeyPoint(x.pt[0], x.pt[1], x.size, ang) for x, ang in zip(kpts,ori)]
    descs = extract_descriptors(kpts, img, As, hardnet, dev)
    return kpts, descs, As

if __name__ == '__main__':
    from matcher import bf_match, fl_match

    # dev = torch.device('cpu')
    dev = torch.device('cuda')
    img1 = cv2.cvtColor(cv2.imread('img_dir/s0.png'), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread('img_dir/200-1.jpg'), cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, [512, 512])
    # Hgt = np.loadtxt('H1to6p')

    NFEATS = 5000
    kpts1, descs1, As1 = detect_sift_HardNet(img1, NFEATS, dev)
    kpts2, descs2, As2 = detect_sift_HardNet(img2, NFEATS, dev)


    matches = bf_match(kp1=kpts1, de1=descs1.astype(np.float32), kp2=kpts2, de2=descs2.astype(np.float32))
    print(len(matches))

    # print(descs1.shape)
    # print(descs2.shape)
    img_matches_bf = cv2.drawMatches(img1, kpts1, img2, kpts2, matches, None)
    cv2.imwrite("hardnet_bf.png", img_matches_bf)
    
    #
    # matches, matchesMask = fl_match(de1=descs1.astype(np.float32), de2=descs2.astype(np.float32))
    # print(len(matches))
    #
    # draw_params = dict(matchesMask=matchesMask, flags=0)
    # img_matches_fl = cv2.drawMatchesKnn(img1, kpts1, img2, kpts2, matches, None, **draw_params)
    # cv2.imwrite("hardnet_fl.png", img_matches_fl)

    # plt.figure(0), plt.imshow(img_matches_32), plt.show()

    # count = bf_match(de1=descs1.astype(np.float32), de2=descs2.astype(np.float32),distance=cv2.NORM_L2)
    # print(count)
    # print(kpts1,descs1)