import torch
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
import os
from torchvision import transforms
import cv2
import numpy
from matplotlib import pyplot as plt
import argparse
import random
random.seed(2023)


parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--demo', default=False, action='store_true')
args = parser.parse_args()


def IoU(box1, box2):
    # box = (x1, y1, x2, y2)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    iou = inter / (box1_area + box2_area - inter)
    return iou

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

def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":numpy.array(['42']),
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
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


# Register refcoco task
tasks.register_task('refcoco', RefcocoTask)
use_cuda = torch.cuda.is_available()
overrides={"bpe_dir":"utils/BPE"}

model_path = '' # Model path

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
    numpy.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

# Move models to GPU
for model in models:
    model.eval()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)

# Image transform
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

# Construct input for refcoco task
patch_image_size = cfg.task.patch_image_size


# for demonstration

if args.demo:
    test_path = '' # path to `.pth` test file
    answer = torch.load(test_path)

    q = 4
    image = Image.open('/{}'.format(answer[q][0])) #INSERT your test image path (provided in github)
    text = answer[q][2]
    gt_box = answer[q][3]

    sample = construct_sample(image, text)
    sample = utils.move_to_cuda(sample) if use_cuda else sample

    # Run eval step for refcoco
    with torch.no_grad():
        result, scores = eval_step(task, generator, models, sample)


    img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.rectangle(
        img,
        (int(result[0]["box"][0]), int(result[0]["box"][1])),
        (int(result[0]["box"][2]), int(result[0]["box"][3])),
        (0, 255, 0),
        3
    )

    cv2.rectangle(
        img,
        (int(gt_box[0]), int(gt_box[1])),
        (int(gt_box[2]), int(gt_box[3])),
        (0, 0, 0),
        1
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    print(answer[q][0])
    print(text)


else:

    testsplits = ['heterogeneous', 'homogeneous', 'cluttered', 'paraphrased']

    for split in testsplits:
        #for original split
        test_path = '{}.pth'.format(split) # INSERT your test data annotation path
        answer = torch.load(test_path)

        yes_cnt_5 = 0
        yes_cnt_8 = 0

        for i, anw in enumerate(answer):
            image = Image.open('{}/{}'.format(split,anw[0])) # INSERT your test data image path
            text = anw[2] 

            # Construct input sample & preprocess for GPU if cuda available
            sample = construct_sample(image, text)
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            with torch.no_grad():
                result, scores = eval_step(task, generator, models, sample)
            if IoU(result[0]['box'], anw[3]) > 0.5:
                yes_cnt_5 += 1
            if IoU(result[0]['box'], anw[3]) > 0.8:
                yes_cnt_8 += 1
        print("{} (IoU 0.5): {:.1f}".format(split,100*yes_cnt_5/(i)))
        print("{} (IoU 0.8): {:.1f}".format(split,100*yes_cnt_8/i))

