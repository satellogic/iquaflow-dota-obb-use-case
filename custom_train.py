import argparse
import os
import mlflow
import glob
import json

def log_mb(trainds):

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

	glob_crit = (
		os.path.join(os.path.basename(trainds),'images','*')
		if not os.path.isdir(os.path.join(os.path.basename(trainds),'images_compressed','*'))
		else os.path.join(os.path.basename(trainds),'images_compressed','*')
		)
			
	size = get_avg_file_size(glob_crit)

	mlflow.log_metric('Mb',size)

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
	logjsonfn = glob.glob(os.path.join(outputpath,'*.json'))[0]

	with open(logjsonfn) as f:
		line_lst = f.readlines()

	lst_dict = [json.loads(line) for line in line_lst][1::] # first dict is always different

	# dict_lst = {k: [dic[k] for dic in lst_dict[1:]] for k in lst_dict[-1]}
	# dict_lst = {
	# 	k: dict_lst[k]
	# 	for k in dict_lst
	# 	if all([
	# 		any([
	# 			isinstance(el,int),
	# 			isinstance(el,float)
	# 		]) 
	# 		for el in dict_lst[k]
	# 	])
	# }

	for eldict in lst_dict:

		mode = eldict['mode']

		eldict2 = {
			k+'_'+mode:float(eldict[k])
			for k in eldict
			if any([isinstance(eldict[k],int),isinstance(eldict[k],float)])
		}
		
		mlflow.log_metrics( eldict2 )

def init_mlf(opt):

	mlflow.set_tracking_uri(opt.mlfuri)

	mlflow.start_run(
		run_id=opt.mlfrunid,
		experiment_id=opt.mlfexpid
	)

	mlflow.log_param('model',opt.model)
	mlflow.log_param('seed',opt.seed)
	mlflow.log_param('dataset',opt.ds)

def edit_dataset_config_file(
	trainds,
	valds,
	testds,
	outputpath
	):
	# edit config file:
	with open('configs/obb/_base_/datasets/dota_template.py') as f:
		txt = f.read()
	
	data_root = os.path.dirname(trainds) + os.sep
	train_ann = os.path.join(os.path.basename(trainds),'annfiles')
	train_img = os.path.join(os.path.basename(trainds),'images')
	val_ann = os.path.join(os.path.basename(valds),'annfiles')
	val_img = os.path.join(os.path.basename(valds),'images')
	test_ann = os.path.join(os.path.basename(testds),'annfiles')
	test_img = os.path.join(os.path.basename(testds),'images')

	txt = txt.replace(r"'__DATA_ROOT__'",f"'{data_root}'")
	txt = txt.replace(r"'__TRAIN_ANNFILES__'",f"'{train_ann}'")
	txt = txt.replace(r"'__TRAIN_IMAGES__'",f"'{train_img}'")
	txt = txt.replace(r"'__VAL_ANNFILES__'",f"'{val_ann}'")
	txt = txt.replace(r"'__VAL_IMAGES__'",f"'{val_img}'")
	txt = txt.replace(r"'__TEST_ANNFILES__'",f"'{test_ann}'")
	txt = txt.replace(r"'__TEST_IMAGES__'",f"'{test_img}'")

	with open('configs/obb/_base_/datasets/dota.py','w') as f:
		f.write(txt)

def get_cmd(dict_args):

	outputpath   = dict_args['outputpath']
	cuda_vis_dev = dict_args['cuda_vis_dev']
	model        = dict_args['model']
	ngpu         = dict_args['ngpu']
	seed         = dict_args['seed']
	ds           = dict_args['ds']

	if model=='fcos':
		cnf_prefix = 'fcos_obb_r50_caffe_fpn_gn-head_4x4_'
		cnf_sufix = ('1x_dota10' if ds=='dota10' else '3x_hrsc')
		subfold = 	'obb/fcos_obb'
	if model=='rcnn':
		cnf_prefix = 'faster_rcnn_obb_r50_fpn_'
		cnf_sufix = ('1x_dota10' if ds=='dota10' else '3x_hrsc')
		subfold = 'obb/faster_rcnn_obb'
	if model=='roitrans':
		cnf_prefix = 'faster_rcnn_roitrans_r50_fpn_'
		cnf_sufix = ('1x_dota10' if ds=='dota10' else '3x_hrsc')
		subfold = 'obb/roi_transformer'

	config = cnf_prefix + cnf_sufix

	if not os.path.isfile(f'configs/{subfold}/{config}.py'):
		print(f'File <{config}.py> does not exist, SKIPING...')
		raise
	
	train_cmd = f"export MKL_THREADING_LAYER=GNU && "
	train_cmd += f"export CUDA_VISIBLE_DEVICES={cuda_vis_dev} && "
	train_cmd += f'./tools/dist_train.sh configs/{subfold}/{config}.py {ngpu} --work-dir {outputpath} --seed {seed} --deterministic'

	map_fn = os.path.join(outputpath,'mAP.txt') 
	chkpnt = os.path.join(outputpath,'latest.pth') 
	# test_cmd = f'python tools/test.py configs/{subfold}/{config}.py '
	# test_cmd += f'{chkpnt} --eval mAP > {map_fn}'

	test_cmd = f"export MKL_THREADING_LAYER=GNU && "
	test_cmd+= f"export CUDA_VISIBLE_DEVICES={cuda_vis_dev} && "
	test_cmd+= f'./tools/dist_test.sh configs/{subfold}/{config}.py {chkpnt} {ngpu} --eval mAP > {map_fn}'

	#bash rather than shell
	# train_cmd = f'/bin/bash -c "{train_cmd}"'
	# test_cmd = f'/bin/bash -c "{test_cmd}"'

	return train_cmd, test_cmd

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	# Define some defaults
	trainds_default     = (os.environ["SM_CHANNEL_TRAINDS"] if "SM_CHANNEL_TRAINDS" in os.environ else "")
	valds_default      = (os.environ["SM_CHANNEL_VALDS"] if "SM_CHANNEL_VALDS" in os.environ else "")
	testds_default      = (os.environ["SM_CHANNEL_TESTDS"] if "SM_CHANNEL_TESTDS" in os.environ else "")
	outputpath_default = (os.environ["SM_OUTPUT_DATA_DIR"] if "SM_OUTPUT_DATA_DIR" in os.environ else "./output")

	# IQF arguments
	parser.add_argument("--trainds", default=trainds_default, type=str, help="training dataset path")
	parser.add_argument("--valds", default=valds_default, type=str, help="validation dataset path")
	parser.add_argument("--testds", default=testds_default, type=str, help="test dataset path")
	parser.add_argument("--outputpath", default=outputpath_default, type=str, help="path output")
	parser.add_argument("--mlfuri", type=str, default="")
	parser.add_argument("--mlfexpid", type=str, default="")
	parser.add_argument("--mlfrunid", type=str, default="")

	parser.add_argument('--cu', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
	parser.add_argument('--model', type=str, default='fcos', help='model')
	parser.add_argument('--ds', type=str, default='dota10', help='dataset')
	parser.add_argument('--seed', type=int, default=123, help='seed')

	opt = parser.parse_args()

	testds, valds, trainds, outputpath = opt.testds, opt.valds, opt.trainds, opt.outputpath

	init_mlf(opt)

	edit_dataset_config_file(
		trainds,
		valds,
		testds,
		outputpath
		)

	train_cmd, test_cmd = get_cmd({
		'outputpath':outputpath,
		'cuda_vis_dev':opt.cu,
		'model':opt.model,
		'ngpu':len(opt.cu.split(',')),
		'seed':opt.seed,
		'ds':opt.ds
	})

	print('========================================================================================')
	print(os.getcwd())
	print(train_cmd+'\n\n')
	print(test_cmd+'\n')
	print('========================================================================================')

	os.system(train_cmd)
	os.system(test_cmd)

	log_mb(trainds)
	log_loss_metrics(outputpath)
	log_ar_map(outputpath)
	
