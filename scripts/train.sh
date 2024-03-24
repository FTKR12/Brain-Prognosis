CUDA_VISIBLE_DEVICES=1 \
python train.py \
    --name test \
    --synthesize_model resvit \
    --dataset_dir /mnt/strokeapp/Datasets/Seg_ctmri_3d \
    --ans_path /home/takuro/callisto/Brain-Prognosis/info.csv \
    --model_name densenet \
    --epochs 100 \
    --train_batch_size 4 \
    --eval_batch_size 20 \
    --lr 1e-4 \
    --weight_decay 1e-5 \