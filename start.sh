# Install BboxToolkit
cd OBBDetection/BboxToolkit/ && \
pip install -v -e . && \
cd ..

# Install OBBDetection
pip install -v -e .

# Install DOTA_devkit
cd DOTA_devkit/ && \
apt-get update && \
apt-get install swig -y  && \
swig -c++ -python polyiou.i && \
python setup.py build_ext --inplace && \
cd ..

# # # Symlinks
# python -c "import os; os.makedirs('data/DOTA1_0',exist_ok=True)" && \
# python -c "import os; os.makedirs('data/split_ss_dota1_0',exist_ok=True)" && \
# ln -sf /share/pgalles/data/DOTA1_0/train $(pwd)/data/DOTA1_0/train && \
# ln -sf /share/pgalles/data/HRSC2016 $(pwd)/data

# # Split val to val/test through symlinks
# export VALDIR=/share/pgalles/data/DOTA1_0/val && python -c "
# import os
# import sys
# import random
# import glob
# random.seed(222)
# val_dir = os.environ['VALDIR']
# fn_lst = glob.glob(os.path.join(val_dir,'images','*'))
# random.shuffle( fn_lst )
# mid = len(fn_lst)//2
# fn_lst_for_test = fn_lst[:mid]
# fn_lst_for_val = fn_lst[mid:]
# for partition,fn_lst in zip(['test','val'],[fn_lst_for_test,fn_lst_for_val]):
#     dst_dir = os.path.join(os.getcwd(),'data/DOTA1_0', partition)
#     os.makedirs(os.path.join(dst_dir,'images'),exist_ok=True)
#     os.makedirs(os.path.join(dst_dir,'labelTxt'),exist_ok=True)
#     for imgfn in fn_lst:
#         dst = os.path.join(dst_dir,'images',os.path.basename(imgfn))
#         os.symlink(imgfn,dst)
#         annfn = imgfn.replace('images','labelTxt').replace('.png','.txt')
#         dst = os.path.join(dst_dir,'labelTxt',os.path.basename(annfn))
#         os.symlink(annfn,dst)
# "

# # Split data
# python BboxToolkit/tools/img_split.py --base_json BboxToolkit/tools/split_configs/dota1_0/ss_train.json
# python BboxToolkit/tools/img_split.py --base_json BboxToolkit/tools/split_configs/dota1_0/ss_val.json
# python BboxToolkit/tools/img_split.py --base_json BboxToolkit/tools/split_configs/dota1_0/ss_test.json
