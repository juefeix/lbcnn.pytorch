
export CUDA_VISIBLE_DEVICES=$1

echo "Using gpus $CUDA_VISIBLE_DEVICES"

python main.py
