#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/1/27
# @Author : zengwb


import os
import numpy as np
import cv2
import argparse
import time
from torch.backends import cudnn
import sys
sys.path.append('..')
import math
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.utils.file_io import PathManager

from predictor import FeatureExtractionDemo
import shutil
from sklearn.metrics.pairwise import cosine_similarity


base =((os.path.abspath('model_yaml'))+'\\') .replace('fast_reid\\demo\\','')

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        default=base+'config-test.yaml',
        help="path to config file",
    )
    parser.add_argument(
        "--parallel",
        action='store_true',
        help='If use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--input",
        # nargs="+",
        default='ORIG_FIGURES/*.JPG',
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=['MODEL.WEIGHTS', base+'model_final.pth'],
        nargs=argparse.REMAINDER,
    )
    return parser


class Reid_feature():
    def __init__(self):
        args = get_parser().parse_args()
        cfg = setup_cfg(args)
        self.demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    def __call__(self, img_list):
        import time
        t1 = time.time()
        feat = self.demo.run_on_image(img_list)
        # print('reid time:', time.time() - t1, len(img_list))
        return feat


def cosin_metric(x1, x2):   # (4, 512)*(512,)
    return np.dot(x1, x2) / (np.linalg.norm(x1, axis=1) * np.linalg.norm(x2))

tar_id = input('请输入您的追踪目标')
# tar_name = input('请输入追踪名称')
# fea_img = r'.\fast_reid\query\Features\{}_features'.format(tar_id)




Reid_feature = Reid_feature()
embs = []
names = []
embs = np.ones((1, 512), dtype=np.int)
# print(q, q.shape)
for person_name in os.listdir('../query/Features/'):
    if person_name == '{}_features'.format(tar_id):
        # if not os.path.isdir(os.path.join('../query', person_name)):
        #     continue
        for image_name in os.listdir(os.path.join('../query/Features/', person_name)):
            img = cv2.imread(os.path.join('../query/Features/', person_name, image_name))

            t1 = time.time()
            feat = Reid_feature(img)   #normlized feat
            # print('pytorch time:', time.time() - t1)
            pytorch_output = feat.numpy()
            embs = np.concatenate((pytorch_output, embs), axis=0)
            # embs.append(pytorch_output)
            names.append(person_name)
            # print(embs.shape, names)
            # print('====sim:', sim)
names = names[::-1]
names.append("None")
print(embs[:-1, :].shape, names)

# print(query.shape)
np.save(os.path.join('../query', 'query_features'), embs[:-1, :])
np.save(os.path.join('../query', 'names'), names)  # save query
def setDir(filepath):
    '''

    如果文件夹不存在就创建，如果文件存在就清空！
    :param filepath:需要创建的文件夹路径
    :return:
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)
feautre_file = '../query/Features'
setDir(feautre_file)
# t1 = time.time()
# query = np.load('../query/query_features.npy')
# cos_sim = cosine_similarity(embs, query)
# print(cos_sim)
# max_idx = np.argmax(cos_sim, axis=1)
# maximum = np.max(cos_sim, axis=1)
# max_idx[maximum < 0.6] = -1
# score = maximum
# results = max_idx
# print(score, results)
# for i in range(4):
#     label = names[results[i]]
#     print(label)
# sim = cosin_metric(embs, q)

# # 文件夹改名
# for name in os.listdir('../query/'):
#     if name[-3:] != 'npy':
#         if name[-3:] != 'res':
#                 ID_name = name

# os.rename('../query/Features/{}_features'.format(tar_id),os.path.join('../query/Features/',tar_name))
#
# shutil.rmtree(os.path.join('../query/',tar_name))
# # pic_names = os.listdir('../query/Features/{}_features'.format(tar_id))
# os.makedirs('../query/{}'.format(tar_name))
# # for pic_name in pic_names:
#     print(pic_name)
#     shutil.copy(('../query/Features/{}_features/'.format(tar_id)+pic_name),'../query/{}'.format(tar_name))
#     # np.testing.assert_allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-6)
#     # print("恭喜你 ^^ ，onnx 和 pytorch 结果一致 ， Exported model has been executed decimal=5 and the result looks good!")
#
#     # np.save(os.path.join(args.output, path.replace('.jpg', '.npy').split('/')[-1]), feat)

