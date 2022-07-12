#! /opt/conda/envs/iqf/bin/python

import sys
import os

#sys.path.append('OBBDetection')
os.chdir('OBBDetection')

from iquaflow.datasets import DSWrapper
from iquaflow.experiments import ExperimentSetup
from iquaflow.experiments.task_execution import PythonScriptTaskExecution

from custom_iqf import (
    DSModifierResize,
    DSModifier_jpg,
    DSModifier_quant
    )

experiment = ExperimentSetup(
    experiment_name         = "iq-dota-obb-use-case",
    task_instance           = PythonScriptTaskExecution(
        model_script_path = '../custom_train.py' ,
        tmp_dir='../remove'
        ),
    ref_dsw_train           = DSWrapper(
        data_path=f'data/split_ss_dota1_0/train',
        data_input=f'data/split_ss_dota1_0/train/images',
        mask_annotations_dir=os.path.join('data/split_ss_dota1_0/train','annfiles')
        ),
    ref_dsw_val             = DSWrapper(
        data_path=f'data/split_ss_dota1_0/val',
        data_input=f'data/split_ss_dota1_0/val/images',
        mask_annotations_dir=os.path.join('data/split_ss_dota1_0/val','annfiles')
        ),
    ref_dsw_test             = DSWrapper(
        data_path=f'data/split_ss_dota1_0/test',
        data_input=f'data/split_ss_dota1_0/test/images',
        mask_annotations_dir=os.path.join('data/split_ss_dota1_0/test','annfiles')
        ),
    ds_modifiers_list       = [
        # DSModifier_jpg(params={"quality": quality}) for quality in range(90,101,2)
    ] + [
        DSModifier_jpg(params={"quality": quality}) for quality in range(10,100,10)
    ] + [
        DSModifier_quant(params={"bits": bits}) for bits in range(1,9)
    ] + [
        DSModifierResize(params={"scaleperc": perc}) for perc in range(10,110,10)
    ],
    repetitions             = 1,
    mlflow_monitoring       = True,
    cloud_options           = {
        'tracking_uri':'file:///work/mlruns'
    },
    extra_train_params      = {
        'cu':['0,1,2,3'],
        'ds': ['dota10'],
        'model':[
            'fcos',
            'rcnn',
            # 'roitrans'
            ],
        'seed': [98],
        }
)

#Execute the experiment
experiment.execute()
