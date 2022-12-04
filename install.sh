conda create -n unip python=3.7
conda activate unip
# torch
pip3 install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# apex
git clone git@github.com:NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
# others
pip install -r requirement.txt
pip install setuptools==58.0.4
# install pycocoeval package from caption decoder

# download ckpt and data
./uni-perceiver-base-L12-H768-224size-pretrained.pth
./small_source_dataset.zip



# get encoder ckpts
export DATA_PATH='/home/kevin/kevin/Uni-Perceiver/small_source_dataset' 
CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch --nproc_per_node=2 \
 main.py --num-gpus 2 \
 --config-file configs/BERT_L12_H768_experiments/zeroshot_config/mscoco_caption.yaml \
 --init_method pytorch --eval-only \
 MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth \
 OUTPUT_DIR work_dirs/experiments

# run image encode
python image_encoder.py --num-gpus 2 \
 --config-file configs/BERT_L12_H768_experiments/zeroshot_config/mscoco_caption.yaml \
 --init_method pytorch --eval-only \
 MODEL.WEIGHTS work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth \
 OUTPUT_DIR work_dirs/experiments