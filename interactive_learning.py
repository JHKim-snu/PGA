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


parser = argparse.ArgumentParser(description='interactive learning')
parser.add_argument('--demo', default=False, action='store_true')
args = parser.parse_args()

#############################
# Load OFA_GVCCI
#############################

tasks.register_task('refcoco', RefcocoTask)
use_cuda = torch.cuda.is_available()
overrides={"bpe_dir":"/home/jhkim/icra24/OFA/utils/BPE"}

ofa_path = '/data/jhkim/iros23/OFA_checkpoints/refcoco_large_best.pt'
ofa_gvcci_path = '/data/jhkim/iros23/OFA_refcoco_checkpoints_0208_pick/0208_train_135/checkpoint_last.pt'

models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(ofa_gvcci_path),
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
# Client: Save raw images (n-images per object, object{}_{}.png)
# Server: Save {obj_id}, {ofa_input} and {personal_q}
#############################

raw_img_path = '/data/jhkim/icra24/raw_images/interaction/'
annotation_path = '/home/jhkim/icra24/interactive/'

annotation = {}

object_cnt = 0

while True: # for all objects
    try:
        while True: # for each object, do until success
            print('---------------------------------')

            is_obj_num_correct = input("is object number {} correct? [y/n] ... ".format(object_cnt))
            if is_obj_num_correct == 'n':
                object_cnt = int(input("insert object num ... "))
            
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
            save_image_path = raw_img_path + '{}.png'.format(object_cnt)
            cv2.imwrite(save_image_path, cv2_img)
            print("saving image to {} ... \n".format(save_image_path))
            
            
            #############################
            ###############vvvvvvvvvvvvvvvv################

            image = Image.open(save_image_path)
            # crop image
            y_crop = 200
            image = image.crop((0,y_crop,image.size[0],image.size[1]))
            # image = np.asarray(image)        
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            correct_directive = 'n'
            while correct_directive == 'n':
                directive = input("Insert the directive ... ") # object in front is my phone
                correct_directive = input("any key if the directive is correct ... if not, 'n' ... ")
            print('\n')
            
            try:
                ofa_input = directive.split("/")[0].strip()
                personal_q = directive.split("/")[1].strip()
            except:
                print('wrong query type... insert "/" between the indicators!!')
                directive = input()
                ofa_input = directive.split("/")[0].strip()
                personal_q = directive.split("/")[1].strip()

            print("object num: {}".format(object_cnt))
            print("ofa_input: {}".format(ofa_input))
            print("personal q: {}".format(personal_q))

            correct_annotation = input("is the info correct? y or n ...")
            if correct_annotation == 'n':
                object_cnt = int(input("insert object count ..."))
                ofa_input = input("insert ofa input ...")
                personal_q = input("insert psersonal indicator ...")
            print('\n')

            annotation['{}'.format(int(object_cnt))] = [ofa_input, personal_q]

            # Construct input sample & preprocess for GPU if cuda available
            sample = construct_sample(image, ofa_input)
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            # Run eval step for refcoco
            print('\n')
            with torch.no_grad():
                result, scores = eval_step(task, generator, models, sample)
                print("OFA inferring with ...'{}'\n".format(ofa_input))

            # Tmp bbox info
            xtl_pick, ytl_pick, xbr_pick, ybr_pick = result[0]["box"][0], result[0]["box"][1]+y_crop, result[0]["box"][2], result[0]["box"][3]+y_crop

            bbox_info = f'{xtl_pick};{ytl_pick};{xbr_pick};{ybr_pick}'
            print(f'Send {bbox_info}\n')
            cli_sock.send(bbox_info.encode())

            if any(xx<0 for xx in [xtl_pick, ytl_pick, xbr_pick, ybr_pick]):
                print("failed... box out of range!\n ofa_input: {}\n retrying...".format(ofa_input))
                continue
            
            success = input("y if success n if fail ... ")
            if success == 'n':
                print("failed... robotic manipulation! retrying ... \n")
                continue

            view = 0
            for k in range(2):
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
                save_image_path = raw_img_path + '{}_{}.png'.format(object_cnt, view)
                cv2.imwrite(save_image_path, cv2_img)
                print("saving image to {} ... \n".format(save_image_path))
                view += 1
                cli_sock.send('give me more!'.encode())

            # cli_sock.send(bbox_info.encode())
            success = input("y if success n if fail ... ")
            if success == 'n':
                print("failed... robotic manipulation! retrying ... \n")
                continue
            
            with open(os.path.join(annotation_path,"interactive{}.json".format(object_cnt)), 'w') as f:
                json.dump(annotation, f)

            object_cnt += 1
            break

        ###############AAAAAAAAAAAAAAAAAA################
        #############################

    except KeyboardInterrupt:
        print('\n Server Ctrl-c')
        break
    except ValueError:
        print('\n Client Closed')
        break

    done = input("done? [y/n]... ")
    if done == 'y':
        break
    else:
        continue




cli_sock.close()
srv_sock.close()

