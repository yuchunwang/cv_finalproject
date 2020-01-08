"""Test PFE on LFW.
"""
# MIT License
# 
# Copyright (c) 2019 Yichun Shi
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import imp
import argparse
import time
import math
import numpy as np
import random
from os import listdir

from utils import utils
from utils.imageprocessing_ytf import preprocess
from utils.dataset import Dataset
from network import Network
from evaluation.ytf import YTFTest


def main(args):


    paths = Dataset(args.dataset_path)['abspath']
    print('%d images to load.' % len(paths))
    assert(len(paths)>0)

    all_path = []
    for j,image_dir in enumerate(paths):
        image_path = listdir(image_dir)
        #one_image = random.choice(image_path)
        one_image = image_path[1]
        one_image = os.path.join(image_dir,one_image)
        all_path.append(one_image)
#         sec_image = random.choice(image_path)
#         sec_image = os.path.join(image_dir,sec_image)
#         all_path.append(sec_image)
            
    # Load model files and config file
    network = Network()
    network.load_model(args.model_dir) 
    images = preprocess(all_path, network.config, False)

    # Run forward pass to calculate embeddings
    mu, sigma_sq = network.extract_feature(images, args.batch_size, verbose=True)
    feat_pfe = np.concatenate([mu, sigma_sq], axis=1)
    
    ytftest = YTFTest(all_path)
    ytftest.init_standard_proto(args.protocol_path)

    accuracy, threshold = ytftest.test_standard_proto(mu, utils.pair_euc_score)
    print('Euclidean (cosine) accuracy: %.5f threshold: %.5f' % (accuracy, threshold))
    accuracy, threshold = ytftest.test_standard_proto(feat_pfe, utils.pair_MLS_score)
    print('MLS accuracy: %.5f threshold: %.5f' % (accuracy, threshold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", help="The path to the pre-trained model directory",
                        type=str)
    parser.add_argument("--dataset_path", help="The path to the YTF dataset directory",
                        type=str, default='data/lfw_mtcnncaffe_aligned')
    # paths = '/data-disk/Jean/aligned_images_DB'
    parser.add_argument("--protocol_path", help="The path to the YTF protocol file",
                        type=str, default='./proto/lfw_pairs.txt')
    #ytf_pairs_file = '/home/jean/Probabilistic_Face_Embeddings/proto/ytf_paris.txt'
    parser.add_argument("--batch_size", help="Number of images per mini batch",
                        type=int, default=128)
    args = parser.parse_args()
    main(args)