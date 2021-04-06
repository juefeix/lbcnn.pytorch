
export CUDA_VISIBLE_DEVICES=$1

IMAGENET_DIR='./data/imagenet/'

echo "Using gpus $CUDA_VISIBLE_DEVICES"

MODEL=$2
BATCH_SIZE=$3

# ps aux | grep zcq | awk '{print $2}' | xargs kill 

PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# # training
# python main.py -a resnet18 ${IMAGENET_DIR}

python main.py -a $MODEL \
    --dist-url "tcp://127.0.0.1:$PORT" \
    --dist-backend 'nccl' \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    --resume checkpoint.pth.tar \
    -b $BATCH_SIZE \
    $IMAGENET_DIR
    # 2>&1 | tee $MODEL.log

cp checkpoint/ckpt.pth checkpoint/$MODEL.ckpt.pth

