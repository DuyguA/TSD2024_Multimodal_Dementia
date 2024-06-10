BATCH_SIZE=64
VAL_BATCH_SIZE=32
EPOCHS=30
GPUS="0"
LR=3e-5 
NFINETUNE=8


python3 -u train_enc.py  --gpu=$GPUS --batch_size=$BATCH_SIZE --val_batch_size=$VAL_BATCH_SIZE --epochs=$EPOCHS --lr=$LR    --nfinetune=$NFINETUNE

