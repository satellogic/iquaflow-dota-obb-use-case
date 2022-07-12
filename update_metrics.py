#!/opt/conda/envs/iqf/bin/python

import os
import glob
import mlflow
import json

from iquaflow.datasets import DSWrapper

from custom_iqf import (
    DSModifierResize,
    DSModifier_jpg,
    DSModifier_quant
    )

def log_ar_map(outputpath):

    map_fn = os.path.join(outputpath,'mAP.txt')

    with open(map_fn) as f:
        txt = f.read()

    txt_1=txt.replace(' ','').split('|class|gts|dets|recall|ap|')[-1]
    txt_2 = [el for el in txt_1.split('\n') if '|' in el]
    # R
    txt_r = [el.split('|')[-3] for el in txt_2]
    recall_lst=[float(el) for el in txt_r if el]
    s = 0
    for r in recall_lst:
        s+=r
    avg_r = s/len(recall_lst)
    # P
    txt_p = [el.replace(' ','').split('|')[-2] for el in txt_2]
    precision_lst=[float(el) for el in txt_p if el] # last one is the avg already

    AR,mAP = avg_r, precision_lst[-1]

    mlflow.log_metric('AR_test',AR)
    mlflow.log_metric('mAP_test',mAP)

def log_loss_metrics(outputpath):
    # read from outputpath, we expect a __.log.json
    try:
        logjsonfn = glob.glob(os.path.join(outputpath,'*.json'))[0]
    except:
        import pdb; pdb.set_trace()

    with open(logjsonfn) as f:
        line_lst = f.readlines()

    lst_dict = [json.loads(line) for line in line_lst][1::] # first dict is always different

    for eldict in lst_dict:

        mode = eldict['mode']

        eldict2 = {
            k+'_'+mode:float(eldict[k])
            for k in eldict
            if any([isinstance(eldict[k],int),isinstance(eldict[k],float)])
        }
        
        mlflow.log_metrics( eldict2 )

def get_file_size(filename):
    from pathlib import Path
    return float(Path(filename).stat().st_size) / 1024 / 1024

def get_avg_file_size(glob_crit):
    size_sum = 0
    fnlst = glob.glob(glob_crit)
    for fn in fnlst:
        size = get_file_size(fn)
        size_sum+=size
    return size_sum/len(fnlst)

def make_ds(pathdir):
    train_dsw = DSWrapper(
        data_path=f'/work/OBBDetection/data/split_ss_dota1_0/train',
        data_input=f'/work/OBBDetection/data/split_ss_dota1_0/train/images',
        mask_annotations_dir=os.path.join('/work/OBBDetection/data/split_ss_dota1_0/train','annfiles')
    )
    base = os.path.basename(pathdir)
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
    mod.modify(ds_wrapper=train_dsw)

def rm_dataset(pathdir):
    os.system(f'rm -rf {pathdir}')

def log_mlf(
    pathdir,
    run,
    experiment_id,
    mb = True,
    outformat = True,
    loss_metrics = True,
    ar_map = True,
    ):
    mlflow.end_run()

    mlflow.start_run(
        run_id=run.info.run_id,
        experiment_id=experiment_id
    )

    artifact_uri = run.info.artifact_uri.replace('file://','')

    if len(glob.glob(os.path.join(artifact_uri,'*')))>0:
        artifact_dir_has_content = True
    else:
        artifact_dir_has_content = False
        print('WARNING: Empty artifact folder for this run:') 
        print(run.data)
    
    # =====================================================

    if all((loss_metrics,artifact_dir_has_content)):
        log_loss_metrics(artifact_uri)
    elif all((loss_metrics, not artifact_dir_has_content)):
        print('skiping log_loss_metrics')

    # =====================================================

    if all((ar_map,artifact_dir_has_content)):
        log_ar_map(artifact_uri)
    elif all((loss_metrics, not artifact_dir_has_content)):
        print('skiping log_ar_map')

    # =====================================================

    if any((mb,outformat)):

        must_make_ds = not os.path.isdir(pathdir)
        if must_make_ds:
            print(f'MAKING, does not exist: {pathdir}')
            make_ds(pathdir)

        glob_crit = ( os.path.join(pathdir,'images/*')
            if not os.path.isdir(os.path.join(pathdir,'images_compressed/*'))
            else os.path.join(pathdir,'images_compressed/*'))
        
        if len(glob.glob(glob_crit))<1:
            print(f"MAKING, because: len(glob.glob('{glob_crit}'))<1")
            make_ds(pathdir)

        # =====================================================
        if all((
            outformat,
            not 'outformat' in run.data.params
            )):

            mlflow.log_param(
                'outformat',
                glob.glob(glob_crit)[0].split('.')[-1]
                )
        
        # =====================================================
        if all((mb,not 'Mb' in run.data.metrics)):
            mlflow.log_metric(
                'Mb',
                get_avg_file_size(glob_crit)
                )

        if must_make_ds:
            rm_dataset(pathdir)

if __name__=='__main__':

    experiment_id = '1'
    tracking_uri = 'file:///work/mlruns'
    mlflow.set_tracking_uri( tracking_uri )

    for run in mlflow.search_runs([experiment_id],output_format='list'):

        pathdir = os.path.join(
            '/work/OBBDetection/data/split_ss_dota1_0/',
            run.data.params['ds_name']
        )

        log_mlf(
            pathdir,
            run,
            experiment_id,
            mb = True,
            outformat = True,
            loss_metrics = False,
            ar_map = False
        )

        # pathdir = os.path.join(
        #     '/work/OBBDetection/data/split_ss_dota1_0/',
        #     run.data.params['ds_name']
        # )
        
        # must_make_ds = not os.path.isdir(pathdir)
        # if must_make_ds:
        #     print(f'MAKING, does not exist: {pathdir}')
        #     make_ds(pathdir)

        # glob_crit = ( os.path.join(pathdir,'images/*')
        #     if not os.path.isdir(os.path.join(pathdir,'images_compressed/*'))
        #     else os.path.join(pathdir,'images_compressed/*'))
        
        # if len(glob.glob(glob_crit))<1:
        #     print(f"MAKING, because: len(glob.glob('{glob_crit}'))<1")
        #     make_ds(pathdir)
        #     continue
        
        # log_mlf(glob_crit,run,experiment_id)

        # if must_make_ds:
        #     rm_dataset(pathdir)