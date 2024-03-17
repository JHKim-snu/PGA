#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6050

dataset_name=propagation_ignore_from2_25_seed_888 ##########
log_dir=./logs
save_dir=/data/jhkim/icra24/ofa_checkpoints ################
mkdir -p $log_dir $save_dir

bpe_dir=/home/jhkim/iros23/OFA/utils/BPE
user_dir=/home/jhkim/iros23/OFA/ofa_module

data_dir=/data/jhkim/icra24/ofa_vg_data/ofa_train_tsv/
data=/data/jhkim/icra24/ofa_vg_data/ofa_train_tsv/propagation_ignore_from2_25_seed_888.tsv,/data/jhkim/icra24/ofa_vg_data/ofa_train_tsv/validation.tsv ###############
restore_file=/data/jhkim/iros23/OFA_refcoco_checkpoints_0208_pick/0208_train_135/checkpoint_last.pt ##############
selected_cols=0,4,2,3
task=refcoco
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.05
lr=2e-5
max_epoch=5
warmup_ratio=0.06
batch_size=1
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=512


log_file=${log_dir}/${dataset_name}".log"
save_path=${save_dir}/${dataset_name}/
mkdir -p $save_path

CUDA_VISIBLE_DEVICES=2,3,4,5 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} /home/jhkim/icra24/OFA/train_vg.py \
    $data \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay --lr=${lr} \
    --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
    --log-format=simple --log-interval=10 \
    --fixed-validation-seed=7 \
    --no-epoch-checkpoints --keep-best-checkpoints=1 \
    --save-interval=1 --validate-interval=1 \
    --save-interval-updates=200 --validate-interval-updates=100 \
    --eval-acc \
    --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
    --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --fp16 \
    --fp16-scale-window=512 \
    --num-workers=0 > ${log_file} 2>&1

