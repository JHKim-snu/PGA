
n_shot=1
device=4
# Define the log file
LOG_FILE="label_propagation_0812(6).txt"

# Clean up the log file if it already exists
if [ -f "$LOG_FILE" ]; then
    rm "$LOG_FILE"
fi

# Define the Python script you want to run
PYTHON_SCRIPT="label_propagation.py"

float_args_probing=$(seq 0.1 0.1 0.3)
float_args_vanilla=$(seq 0.6 0.1 0.9)
float_args_full_tuning=$(seq 0.1 0.1 0.3)
sample_n=$(seq 100 100 300)
# nshot_args=$(seq 2 1 5)

# Loop over arguments and call the Python script

# for nshot in $nshot_args; do

# for model in 'linear_probing'; do
#     for thresh in $float_args_probing; do
#         echo "Running with argument: $model $thresh $n_shot shot" | tee -a $LOG_FILE
#         CUDA_VISIBLE_DEVICES=${device} python $PYTHON_SCRIPT --model $model --thresh $thresh --iter 3 --save_nodes False --nshot $n_shot 2>&1 | tee -a $LOG_FILE
#         echo -e "\n" | tee -a $LOG_FILE
#         echo -e "\n" | tee -a $LOG_FILE
#     done
# done

for model in 'vanilla'; do
    for thresh in $float_args_vanilla; do
        echo "Running with argument: $model $thresh" | tee -a $LOG_FILE
        CUDA_VISIBLE_DEVICES=${device} python $PYTHON_SCRIPT --model $model --thresh $thresh --iter 3 --save_nodes True --nshot $n_shot 2>&1 | tee -a $LOG_FILE
        echo -e "\n" | tee -a $LOG_FILE
        echo -e "\n" | tee -a $LOG_FILE
    done
done

# thresh=0.6
# for model in 'vanilla'; do
#     for sample_num in $sample_n; do
#         echo "Running with argument: $model $thresh" | tee -a $LOG_FILE
#         CUDA_VISIBLE_DEVICES=${device} python $PYTHON_SCRIPT --model $model --thresh $thresh --iter 3 --save_nodes True --nshot $n_shot --sample_n $sample_num 2>&1 | tee -a $LOG_FILE
#         echo -e "\n" | tee -a $LOG_FILE
#         echo -e "\n" | tee -a $LOG_FILE
#     done
# done

# for model in 'full_tuning'; do
#     for thresh in $float_args_full_tuning; do
#         echo "Running with argument: $model $thresh $n_shot shot" | tee -a $LOG_FILE
#         CUDA_VISIBLE_DEVICES=${device} python $PYTHON_SCRIPT --model $model --thresh $thresh --iter 3 --save_nodes False --nshot $n_shot 2>&1 | tee -a $LOG_FILE
#         echo -e "\n" | tee -a $LOG_FILE
#         echo -e "\n" | tee -a $LOG_FILE
#     done
# done
# done
