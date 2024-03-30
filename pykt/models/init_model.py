import torch
import numpy as np
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from .gpt4kt import GPT4KT
from collections import OrderedDict

device = "cpu" if not torch.cuda.is_available() else "cuda"

def init_model(model_name, model_config, data_config, emb_type, args=None, num_stu=None, mode="train", train_start=True):
    if model_name == "gpt4kt":
        # 2） 配置每个进程的gpu
        # if mode == "train" and train_start:
        #     print(f"init torch.distributed.init_process_group")
        #     # torch.distributed.init_process_group(backend='nccl')
        #     # torch.cuda.set_device(args.local_rank)
        if emb_type.find("pt") == -1:
            model = GPT4KT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
        else:
            model = GPT4KT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"], num_sgap=data_config["num_sgap"]).to(device)
        print(f"mode:{mode}")
        if mode == "train" and train_start:
            model = DDP(model)
    else:
        print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path, args=None, mode="test", finetune=False):
    model = init_model(model_name, model_config, data_config, emb_type, args, mode=mode)
    net = torch.load(os.path.join(ckpt_path, emb_type+"_model.module.ckpt"),map_location="cpu")
    # print(f"net:{net}")
    if not finetune:
        model.load_state_dict(net)
    else:
        new_state_dict = OrderedDict()
        for k,v in net.items():
            if "module" not in k:
                k = "module." + k
                # k = k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    return model
