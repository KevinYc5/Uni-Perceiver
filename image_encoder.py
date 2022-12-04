import torch
from torch import nn
import numpy as np
from uniperceiver.tokenization import ClipTokenizer
from uniperceiver.modeling.embedding import VideoBaseEmbedding, TokenBaseEmbedding
from uniperceiver.modeling.encoder import UnifiedBertEncoder
from uniperceiver.config import get_cfg, CfgNode
from uniperceiver.modeling import add_config
import os
from torchvision import transforms
import einops
import PIL.Image as Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import copy
# encode the image/video sequence.
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        cfg = get_cfg()
        config_file = "configs/BERT_L12_H768_experiments/zeroshot_config/mscoco_caption.yaml"
        tmp_cfg = cfg.load_from_file_tmp(config_file)
        add_config(cfg, tmp_cfg)
        cfg.merge_from_file(config_file)
        add_data_prefix(cfg)
        opts = ['MODEL.WEIGHTS', 'work_dirs/pretrained_models/uni-perceiver-base-L12-H768-224size-pretrained.pth']
        cfg.merge_from_list(opts)
        add_default_setting_for_multitask_config(cfg)
        cfg.freeze()
        self.video_embed = VideoBaseEmbedding(cfg)
        self.tokenizer = ClipTokenizer()
        self.token_embed = TokenBaseEmbedding(cfg)
        self.fused_encoder = UnifiedBertEncoder(cfg)
        self.image_transform = self.build_image_transform()
        PATH = ["token_embed.pth", "token_embed.pth", "token_embed.pth"]
        for idx, model in enumerate([self.token_embed, self.token_embed, self.token_embed]):
            model.load_state_dict(torch.load(PATH[idx]))
        model.eval()
        self.fp16 = False

    def data_half(self, data):
        if self.fp16:
            if isinstance(data, torch.Tensor) and data.dtype == torch.float32:
                data = data.half()
                    # print(k)
            return data
        else:
            return data


    def get_spe_token(self, tokenizer, token_embed):
        a = torch.tensor(tokenizer.encode('<|spe|>')).unsqueeze(0)  # bs, 1
        return token_embed(a, type_embed=False, pos_embed=False)

    def preprocess(self, tokenizer, token_embed, data):
        # perparation for fused_encoder input
        bs = data.shape[0]
        device =  data.device
        combined_data = []
        # spe embedding
        spe_token = self.get_spe_token(tokenizer, token_embed).expand(bs, -1, -1).to(device)

        print("spe_token: ", spe_token.shape)
        length = [data.shape[1]]
        length = [1] + length
        combined_data.append(spe_token)


        combined_data.append(data)

        combined_data = torch.cat(combined_data, dim=1)

        combined_valid_mask = None
        data_type = 'input'
        sample_info = {}
        moe_embedding = None
        
        return {
            'data': combined_data,
            'invalid_mask': combined_valid_mask,
            'data_type': data_type,
            'sample_info': sample_info,
            'moe_embedding': moe_embedding,
        }

    def _forward_data(self, data, history_states=None, return_all=False):


        data = self.video_embed(data)

        preprocess_data = self.preprocess(self.tokenizer, self.token_embed, data)

        fused_data = self.fused_encoder(**preprocess_data, task_info={'task_type': 'image_caption'}, history_states=history_states, return_all=return_all)
        return fused_data
    
    def build_image_transform(self,input_size=224):
        t = []
        t.append(
            transforms.Resize(input_size, interpolation=3)  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
        return transforms.Compose(t)

    def forward(self, input):
        # input: -1 ~1 256x256
        input = (input + 1.0) * 127.5
        images = []
        for batch_idx in range(input.shape[0]):
            image = einops.rearrange(input[batch_idx], 'c h w -> h w c')
            images.append(self.image_transform(Image.fromarray(image.astype(np.uint8), 'RGB')).unsqueeze(0))
        input = torch.cat(images)
        # print(input.shape) # torch.Size([4, 3, 224, 224])
        input = self.data_half(input)
        image_encode = self._forward_data(input, return_all=True)

        return image_encode

def add_data_prefix(cfg):
    # TODO: more flexible method
    data_dir = os.getenv("DATA_PATH", None)
    mapping_list = [
        [cfg.DATALOADER, 'FEATS_FOLDER', ['DATALOADER',]],
        [cfg.DATALOADER, 'ANNO_FOLDER', ['DATALOADER', ]],
        [cfg.DATALOADER, 'CLASS_NAME_FILE', ['DATALOADER', ]],
        [cfg.INFERENCE, 'VOCAB', ['INFERENCE', ]],
        [cfg.INFERENCE, 'VAL_ANNFILE', ['INFERENCE', ]],
        [cfg.INFERENCE, 'TEST_ANNFILE', ['INFERENCE',]],
        [cfg.MODEL, 'WEIGHTS', ['MODEL',]],
    ]
    whitelist = ["BERT", "CLIP", "CLIP_CAPTION"]
    if data_dir:
        for node, attr ,_ in mapping_list:
            if node[attr] != '' and not node[attr].startswith('.') and not node[attr].startswith('/') and not node[attr].startswith('work_dirs') and not node[attr].startswith('cluster') and not node[attr].startswith('s3://') and node[attr] not in whitelist:
                setattr(node, attr, os.path.join(data_dir, node[attr]))
    for task in cfg.TASKS:
        for _, item, key_list in mapping_list:
            config_tmp = task
            for key in key_list:
                if key in config_tmp:
                    config_tmp = config_tmp[key]
            if item in config_tmp and  config_tmp[item] != '' and not config_tmp[item].startswith('.') and not config_tmp[item].startswith('/') and not config_tmp[item].startswith('work_dirs') and not config_tmp[item].startswith('cluster') and not config_tmp[item].startswith('s3://') and config_tmp[item] not in whitelist:
                config_tmp[item] =  os.path.join(data_dir, config_tmp[item])

    mapping_list = [
        ['', 'FILE_PATH', ['SHARED_TARGETS_CFG',]],
    ]
    if cfg.SHARED_TARGETS is None:
        cfg.SHARED_TARGETS = []
    for share_targets in cfg.SHARED_TARGETS:
        for _, item, key_list in mapping_list:
            config_tmp = share_targets
            for key in key_list:
                config_tmp = config_tmp[key]
            if item in config_tmp and config_tmp[item] != '' and not config_tmp[item].startswith('.') and not config_tmp[item].startswith(
                    '/') and not config_tmp[item].startswith('work_dirs') and not config_tmp[item].startswith(
                        'cluster') and not config_tmp[item].startswith('s3://') and config_tmp[item] not in whitelist:
                config_tmp[item] = os.path.join(data_dir, config_tmp[item])

def add_default_setting_for_multitask_config(cfg):
    # merge some default config in (CfgNode) uniperceiver/config/defaults.py to each task config (dict)

    tasks_config_temp = cfg.TASKS
    num_tasks = len(tasks_config_temp)
    cfg.pop('TASKS', None)

    cfg.TASKS = [copy.deepcopy(cfg) for _ in range(num_tasks)]

    for i, task_config in enumerate(tasks_config_temp):
        cfg.TASKS[i].merge_from_other_cfg(CfgNode(task_config))
        cfg.TASKS[i] = cfg.TASKS[i].to_dict_object()
        pass

def main():

    # build model
    image_encoder = ImageEncoder()
    # get data
    data = np.ones([4, 3, 224, 224]) # [-1, 1]
    # get result
    res = image_encoder(data)
    print(len(res))
    print(res[0].shape)


main()
