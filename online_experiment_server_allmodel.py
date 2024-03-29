import sys
import socket
import cv2
import numpy as np
import base64
import os

import torch
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
import argparse
from torchvision import transforms
import json
import random

#############################
# Load model
#############################

tasks.register_task('refcoco', RefcocoTask)
use_cuda = torch.cuda.is_available()
overrides={"bpe_dir":"/home/jhkim/icra24/OFA/utils/BPE"}

model_path0 = '' # Path to Passive model checkpoint file
model_path1 = '' # Path to PGA model checkpoint file
model_path2 = '' # Path to Supervised model checkpoint file

models0, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path0),
        arg_overrides=overrides
    )
models1, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path1),
        arg_overrides=overrides
    )
models2, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path2),
        arg_overrides=overrides
    )

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 4
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 4
cfg.generation.no_repeat_ngram_size = 3

# Fix seed for stochastic decoding
if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

# Move models to GPU
for model in models0:
    model.eval()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

for model in models1:
    model.eval()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

for model in models2:
    model.eval()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator0 = task.build_generator(models0, cfg.generation)
generator1 = task.build_generator(models1, cfg.generation)
generator2 = task.build_generator(models2, cfg.generation)

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()
def encode_text(text, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text.lower()),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for refcoco task
patch_image_size = cfg.task.patch_image_size
def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "w_resize_ratios": w_resize_ratio,
        "h_resize_ratios": h_resize_ratio,
        "region_coords": torch.randn(1, 4)
    }
    return sample

#############################
# Connect Socket
#############################

HOST = '147.47.200.155'
PORT = 9998

srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv_sock.bind((HOST, PORT))
srv_sock.listen()
cli_sock, addr = srv_sock.accept()
print(f'Connected by: {addr}')


#############################
#############################


#############################
# Receive raw image and save
# Send coordinates
#############################

raw_img_path = '/data/jhkim/icra24/raw_images/online_exp/'
test_annotation_path = '/data/jhkim/icra24/ofa_vg_data/test_annotation'
test_splits = ['seen_random', 'seen_same', 'unseen_random', 'unseen_same']

random.seed(2023)
for split in test_splits:
    globals()['{}_path'.format(split)] = os.path.join(test_annotation_path, split+'.pth')
    tmp = torch.load(globals()['{}_path'.format(split)])
    globals()[f'{split}'] = random.sample(tmp, 15)

trial_cnt = 0

starting_split = input("insert the split name ...")
starting_idx = int(input("insert the staring index ..."))
stay = False

while True: # till Ctrl+c
    for split in test_splits:
        if starting_split != split and stay == False:
            continue
        elif starting_split == split:
            globals()[f'{split}'] = globals()[f'{split}'][starting_idx:]    
            stay = True
        print('---------------------------------')
        print('---------------------------------')
        print('test split: {}'.format(split))
        
        for anw in globals()[f'{split}']:
            for mdl_ind in range(3):
                while True: # do until success = 'y'
                    print('---------------------------------')
                    print('trial {}'.format(trial_cnt))
                    print('model {}'.format(mdl_ind))
                    print('image id: {}'.format(anw[0]))
                    print('personal indicator: {}'.format(anw[2]))

                    # Receive CV2 Image
                    length = int(cli_sock.recv(64).decode('utf-8'))
                    buf = b''
                    while length:
                        newbuf = cli_sock.recv(length)
                        buf += newbuf
                        length -= len(newbuf)
                    print("image recieved from the robot!")
                    
                    data = np.frombuffer(base64.b64decode(buf), np.uint8)
                    cv2_img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                    # save received image
                    save_image_path = raw_img_path + '{}.png'.format(trial_cnt)
                    cv2.imwrite(save_image_path, cv2_img)
                    print("saving image to {} ... \n".format(save_image_path))
                    
                    
                    #############################
                    ###############vvvvvvvvvvvvvvvv################

                    image = Image.open(save_image_path)
                    # crop image
                    # y_crop = 200
                    # image = image.crop((0,y_crop,image.size[0],image.size[1]))
                    # image = np.asarray(image)        
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    text = anw[2]

                    print('\n')

                    # Construct input sample & preprocess for GPU if cuda available
                    sample = construct_sample(image, text)
                    sample = utils.move_to_cuda(sample) if use_cuda else sample

                    # Run eval step for refcoco
                    print('\n')
                    with torch.no_grad():
                        result, scores = eval_step(task, globals()['generator{}'.format(mdl_ind)], globals()['models{}'.format(mdl_ind)], sample)
                        print("OFA inferring with ...'{}'\n".format(text))

                    # Tmp bbox info
                    xtl_pick, ytl_pick, xbr_pick, ybr_pick = result[0]["box"][0], result[0]["box"][1], result[0]["box"][2], result[0]["box"][3]

                    bbox_info = f'{xtl_pick};{ytl_pick};{xbr_pick};{ybr_pick}'
                    print(f'Send {bbox_info}\n')
                    cli_sock.send(bbox_info.encode())
                    
                    redo = input("y if redo, else press any key... ")
                    if redo == 'y':
                        print("redo trial {}... \n".format(trial_cnt))
                        continue

                    trial_cnt += 1
                    break 

                    ###############AAAAAAAAAAAAAAAAAA################
                    #############################

                # except KeyboardInterrupt:
                #     print('\n Server Ctrl-c')
                #     break
                # except ValueError:
                #     print('\n Client Closed')
                #     break



cli_sock.close()
srv_sock.close()

