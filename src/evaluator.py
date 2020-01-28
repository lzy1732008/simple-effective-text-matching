# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from datetime import timedelta
import os
from pprint import pprint
import tensorflow as tf
from .model import Model
from .interface import Interface
from .utils.loader import load_data


class Evaluator:
    def __init__(self, model_path, data_file):
        self.model_path = model_path
        self.data_file = data_file

    def evaluate(self):
        start_time = time.time()
        data = load_data(*os.path.split(self.data_file))

        tf.reset_default_graph()
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.allow_soft_placement = True
            sess = tf.Session(config=config)
            with sess.as_default():
                model, checkpoint = Model.load(sess, self.model_path)
                args = checkpoint['args']
                interface = Interface(args)
                batches = interface.pre_process(data, training=False)
                _, stats = model.evaluate(sess, batches)
                pprint(stats)
                time_dif = get_time_dif(start_time)
                print("Time usage:", time_dif)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
