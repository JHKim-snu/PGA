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
from PIL import Image
import argparse
from torchvision import transforms
import random

parser = argparse.ArgumentParser(description='interactive learning')
parser.add_argument('--demo', default=False, action='store_true')
args = parser.parse_args()

#############################
# Load model
#############################

tasks.register_task('refcoco', RefcocoTask)
use_cuda = torch.cuda.is_available()
overrides={"bpe_dir":"../utils/BPE"}

model_path = '' # INSERT the model path

models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path),
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
for model in models:
    model.eval()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

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

HOST = '' # INSERT HOST
PORT = 9998

srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv_sock.bind((HOST, PORT))
srv_sock.listen()
cli_sock, addr = srv_sock.accept()
print(f'Connected by: {addr}')


#############################
# Receive raw image and save
# Send coordinates
#############################

raw_img_path = '' # path to save images
test_annotation_path = '' # Directory where the test annotations are
test_splits = ['heterogeneous', 'homogeneous', 'cluttered', 'paraphrased']

random.seed(2023)
for split in test_splits:
    globals()['{}_path'.format(split)] = os.path.join(test_annotation_path, split+'.pth')
    tmp = torch.load(globals()['{}_path'.format(split)])
    globals()[f'{split}'] = random.sample(tmp, 15)

trial_cnt = 0

if args.demo:
    while True: # till Ctrl+c
        print('---------------------------------')
        print('---------------------------------')
        try:
            print('---------------------------------')

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
            save_image_path = raw_img_path + '{}_demo.png'.format(trial_cnt)
            cv2.imwrite(save_image_path, cv2_img)
            print("saving image to {} ... \n".format(save_image_path))
            
            #############################
            ###############vvvvvvvvvvvvvvvv################

            image = Image.open(save_image_path)
            text = input("insert the instruction ... ")
            print('\n')

            # Construct input sample & preprocess for GPU if cuda available
            sample = construct_sample(image, text)
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            # Run eval step for refcoco
            print('\n')
            with torch.no_grad():
                result, scores = eval_step(task, generator, models, sample)

            # Tmp bbox info
            xtl_pick, ytl_pick, xbr_pick, ybr_pick = result[0]["box"][0], result[0]["box"][1], result[0]["box"][2], result[0]["box"][3]

            bbox_info = f'{xtl_pick};{ytl_pick};{xbr_pick};{ybr_pick}'
            print(f'Send {bbox_info}\n')
            cli_sock.send(bbox_info.encode())

            trial_cnt += 1

            ###############AAAAAAAAAAAAAAAAAA################
            #############################

        except KeyboardInterrupt:
            print('\n Server Ctrl-c')
            break
        except ValueError:
            print('\n Client Closed')
            break

else:
    while True: # till Ctrl+c
        for split in test_splits:
            print('---------------------------------')
            print('---------------------------------')
            print('test split: {}'.format(split))
            for anw in globals()[f'{split}']:
                try:
                    print('---------------------------------')
                    print('trial {}'.format(trial_cnt))
                    print('image id: {}'.format(anw[0]))

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

                    image = Image.open(save_image_path)
                    text = anw[2]
                    print('\n')

                    # Construct input sample & preprocess for GPU if cuda available
                    sample = construct_sample(image, text)
                    sample = utils.move_to_cuda(sample) if use_cuda else sample

                    # Run eval step for refcoco
                    print('\n')
                    with torch.no_grad():
                        result, scores = eval_step(task, generator, models, sample)
                        print("inferring with ...'{}'\n".format(text))

                    # Tmp bbox info
                    xtl_pick, ytl_pick, xbr_pick, ybr_pick = result[0]["box"][0], result[0]["box"][1], result[0]["box"][2], result[0]["box"][3]

                    bbox_info = f'{xtl_pick};{ytl_pick};{xbr_pick};{ybr_pick}'
                    print(f'Send {bbox_info}\n')
                    cli_sock.send(bbox_info.encode())
                    
                    trial_cnt += 1

                    ###############AAAAAAAAAAAAAAAAAA################
                    #############################

                except KeyboardInterrupt:
                    print('\n Server Ctrl-c')
                    break
                except ValueError:
                    print('\n Client Closed')
                    break



cli_sock.close()
srv_sock.close()

