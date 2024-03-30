import torch
import numpy as np
from fairseq import utils, tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.refcoco import RefcocoTask
from models.ofa import OFAModel
from PIL import Image
import os
import cv2
from matplotlib import pyplot as plt


test_path = ''
answer = torch.load(test_path)

tasks.register_task('refcoco', RefcocoTask)
use_cuda = torch.cuda.is_available()
overrides={"bpe_dir":"utils/BPE"}

model_path1 = ''

models1, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path1),
        arg_overrides=overrides
    )

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 4
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 4
cfg.generation.no_repeat_ngram_size = 3

if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

for model in models1:
    model.eval()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

generator1 = task.build_generator(models1, cfg.generation)

# Image transform
from torchvision import transforms
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

q = 1
image = Image.open('/{}'.format(answer[q][0])) # INSERT the iamge file path
text = answer[q][2]
gt_box = answer[q][3]
sample = construct_sample(image, text)
sample = utils.move_to_cuda(sample) if use_cuda else sample

with torch.no_grad():
    result, scores = eval_step(task, generator1, models1, sample)


img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
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
#print(result[0]["box"])

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
print(answer[q][0])
print(text)
