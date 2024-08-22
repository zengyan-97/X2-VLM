import os
import json


def marvl_preproc(ipath, opath):
    if not os.path.exists(opath):
        os.makedirs(opath)
    # test
    root = os.path.join(ipath, 'zero_shot/annotations')
    for fp in os.listdir(root):
        with open(os.path.join(root, fp)) as f, open(os.path.join(opath, fp[:-1]), 'w') as wf:
            data = []
            for l in f:
                d = json.loads(l)
                data.append({
                    'sentence': d['caption'],
                    'label': d['label'],
                    'images': ['images/marvl_official/{}/images/{}/{}'.format(d['language'], d['left_img'].split('-')[0], d['left_img']),
                               'images/marvl_official/{}/images/{}/{}'.format(d['language'], d['right_img'].split('-')[0], d['right_img'])]
                })
            json.dump(data, wf)
    # few shot
    root = os.path.join(ipath, 'few_shot/annotations')
    for fp in os.listdir(root):
        with open(os.path.join(root, fp)) as f, open(os.path.join(opath, fp[:-1]), 'w') as wf:
            data = []
            for l in f:
                d = json.loads(l)
                data.append({
                    'sentence': d['caption'],
                    'label': d['label'],
                    'images': ['images/marvl_fewshot/{}/all/{}'.format(d['language'], d['left_img'].split('/')[-1]),
                               'images/marvl_fewshot/{}/all/{}'.format(d['language'], d['right_img'].split('/')[-1])]
                })
            json.dump(data, wf)
