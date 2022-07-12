#!/opt/conda/envs/iqf/bin/python

import os
import sys
import glob
import mlflow

from iquaflow.datasets import DSWrapper

from custom_iqf import (
    DSModifierResize,
    DSModifier_jpg,
    DSModifier_quant
    )

def make_ds(pathdir):
    test_dsw = DSWrapper(
        data_path=f'/work/OBBDetection/data/split_ss_dota1_0/test',
        data_input=f'/work/OBBDetection/data/split_ss_dota1_0/test/images',
        mask_annotations_dir=os.path.join('/work/OBBDetection/data/split_ss_dota1_0/test','annfiles')
    )
    base = os.path.basename(pathdir)
    print('pathdir',pathdir,'base',base)
    if 'jpg' in base:
        #jpg modifier
        quality = int(base.split('#')[-1].split('_')[-2].replace('jpg',''))
        mod = DSModifier_jpg(params={"quality": quality})
    elif 'resize' in base:
        scale = int(base.split('#')[-1].replace('resize',''))
        mod = DSModifierResize(params={"scaleperc": scale})
    elif 'quant' in base:
        bits = int(base.split('#')[-1].split('_')[-2].replace('quant',''))
        mod = DSModifier_quant(params={"bits": bits})
    mod.modify(ds_wrapper=test_dsw)
    
if __name__ == "__main__":
    
    make_ds(sys.argv[-1])