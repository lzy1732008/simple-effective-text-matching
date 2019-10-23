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


from zhon.hanzi import punctuation
import os
import json
import string
import numpy as np
import msgpack
import jieba
from collections import Counter

in_dir = 'orig/WSFX'
out_dir = '../models/wsfx'
data_dir = 'wsfx'
label_map = {0: '0', 1: '1'}

def createEnv():
    os.makedirs(out_dir, exist_ok=True)
    #读取vocab.txt文件，建立word_index
    env = {}
    with open(os.path.join(in_dir, 'vocab.txt'),'r', encoding='utf-8') as f:
        lines = f.readlines()
        vocab = [w.split()[0] for w in lines if w.strip() != '']
        env["word_index"] = {w : i + 2 for i, w in enumerate(vocab)}
        env["word_index"]["<pad>"] = 0
        env["word_index"]["<unk>"] = 1

    with open(os.path.join(out_dir, 'vocab.txt'), 'w', encoding='utf-8') as f:
        for w in vocab:
            f.write('{}\n'.format(w))

    #建立train，Dev，test
    train = processInitDataSet(os.path.join(in_dir, 'train-原始.txt'))
    test = processInitDataSet(os.path.join(in_dir, 'test-原始.txt'))
    dev = processInitDataSet(os.path.join(in_dir, 'dev-原始.txt'))
    env['train'] = train
    env['test'] = test
    env['dev'] = dev
    with open(os.path.join(in_dir, 'env'), 'w', encoding='utf-8') as f:
        json.dump(env, f)

def processInitDataSet(inputPath):
    result = []
    with open(inputPath, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line != '':
                items = line.split('|')
                assert len(items) == 4, ValueError("The number of items in this line is less than 4")
                fact = processText(items[1])
                law = processText(items[2])
                assert items[3] in ['0', '1'], ValueError("Label is not in [0,1]!")
                label = int(items[3])
                result.append([fact,law,label])
    return result

def processText(line):
    initContent = line.strip()
    if initContent != "":
        content = jieba.cut(initContent)
        lines = list(map(lambda x: str(x).strip(), content))
        contentcut = list(filter(lambda x: x != "", lines))
        contentcut.insert(0, '<S>')
        contentcut.append('<E>')
        return contentcut
    return []

def createDataSet():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(in_dir, 'env')) as f:
        env = json.load(f)

    # 建立word2idx之间的双向映射，存储在txt文件中
    print('convert_vocab ...')
    w2idx = env['word_index']
    print(len(w2idx))
    assert w2idx["<pad>"] == 0
    assert w2idx["<unk>"] == 1
    idx2w = {i: w for w, i in w2idx.items()}

    # save data files
    # 预处理切分好的数据集，并存储到相应的txt文件中
    punctuactions = set.union(set(string.punctuation), punctuation)
    for split in ['train', 'dev', 'test']:
        labels = Counter()
        print('convert', split, '...')
        data = env[split]
        with open(os.path.join(data_dir, '{}.txt'.format(split)), 'w') as f_out:
            for sample in data:
                a, b, label = sample
                a = a[1:-1]
                b = b[1:-1]
                a = [w for w in a if w and w not in punctuactions]
                b = [w for w in b if w and w not in punctuactions]
                # assert all(w in w2idx for w in a) and all(w in w2idx for w in b)
                a = ' '.join(a)
                b = ' '.join(b)
                assert len(a) != 0 and len(b) != 0
                labels.update({label: 1})
                assert label in label_map
                label = label_map[label]
                f_out.write('{}\t{}\t{}\n'.format(a, b, label))
        print('labels:', labels)

def createMsgpackFile():
    fw = open(os.path.join(out_dir, 'embedding_w2v.msgpack'), 'wb')

    with open(os.path.join(in_dir, 'vectors_w2v.txt'),'r', encoding='utf-8') as f:
         lines = f.readlines()
         vectors = [tuple([0] * 128), tuple([0] * 128)]
         for line in lines:
             line = line.strip()
             if line != "":
                 vector = tuple(line.split()[1:])
                 vectors.append(vector)
         msgpack.dump(vectors, fw)



# createMsgpackFile()
# fr = open(os.path.join(out_dir, 'embedding_w2v.msgpack'), 'rb')
# emb = msgpack.load(fr,encoding = 'utf-8')
# print(len(emb))

