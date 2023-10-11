import argparse

from model.run_train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data_arg
    parser.add_argument(
        "--output_dir", type=str, default="./novelImgcaption"
    )
    parser.add_argument("--dataset_huggingface", type=str, default="minwook/imgKoNovel")
    parser.add_argument(
        "--encoder_model_name_or_path",
        type=str,
        default="ddobokki/vision-encoder-decoder-vit-gpt2-coco-ko",
    )
    parser.add_argument(
        "--decoder_model_name_or_path", type=str, default="skt/kogpt2-base-v2"
    )
    parser.add_argument("--max_length", type=int, default=126)

    # train_arg
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_beams", type=int, default = 100)

    # eval_arg
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--logging_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)

    args = parser.parse_args()
    train(args)