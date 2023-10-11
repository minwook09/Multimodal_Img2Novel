export CUDA_VISIBLE_DEVICES=0

python -u main.py \
    --output_dir='./novelImgcaption'\
    --dataset_huggingface='minwook/imgKoNovel'\
    --encoder_model_name_or_path="ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko"  --decoder_model_name_or_path="skt/kogpt2-base-v2" \
    --seed=42 --num_train_epochs=16 \
    --train_batch_size=32 --val_batch_size=4 \
    --logging_steps=500 --eval_steps=500