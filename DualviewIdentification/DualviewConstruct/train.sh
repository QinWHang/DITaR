export CUDA_VISIBLE_DEVICES=6
DATASET="amazon-beauty"
MODEL_NAME="sasrec"

ALPHA=0.5
LAMBDA_C=0.1
NUM_EPOCHIS=200
HIDDEN_SIZE=64

TRAIN_DATA_PATH="./data/processed/${DATASET}/${DATASET}.inter.txt"
OUTPUT_DIR="./Outputs/dual_view_model_outputs/"



# 模型特定参数
if [ "$MODEL_NAME" == "sasrec" ]; then
    MODEL_PARAMS="--trm_num 2 --num_heads 4 --dropout_rate 0.3"
elif [ "$MODEL_NAME" == "bert4rec" ]; then
    MODEL_PARAMS="--trm_num 2 --num_heads 2 --mask_prob 0.3 --dropout_rate 0.5"
elif [ "$MODEL_NAME" == "gru4rec" ]; then
    MODEL_PARAMS="--num_layers 1 --dropout_rate 0.3"
fi


python ./DualviewIdentification/DualviewConstruct/train_dual_view.py \
    --dataset ${DATASET} \
    --model_name ${MODEL_NAME} \
    --train_data_path ${TRAIN_DATA_PATH} \
    --embedding_dir ./data_new/embeddings \
    --output_dir ${OUTPUT_DIR} \
    --max_len 200 \
    --pca_dim 64 \
    --hidden_size ${HIDDEN_SIZE} \
    --alpha ${ALPHA} \
    --lambda_c ${LAMBDA_C} \
    --num_train_epochs ${NUM_EPOCHIS} \
    --lr 0.001 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --patience 10 \
    --eval_ks 10,20 \
    ${MODEL_PARAMS} \
