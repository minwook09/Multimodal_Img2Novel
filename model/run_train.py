import requests
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= CUDA_VISIBLE_DEVICES
import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    default_data_collator,
)
from datasets import load_dataset
from model.Noveldataset import ImageCaptioningDataset
import numpy as np

from nltk.translate.bleu_score import corpus_bleu


def train(args):
    # device setting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load feature extractor and tokenizer
    encoder_model_name_or_path = args.encoder_model_name_or_path
    feature_extractor = ViTFeatureExtractor.from_pretrained(encoder_model_name_or_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        encoder_model_name_or_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",  
    )

    # load model
    model = VisionEncoderDecoderModel.from_pretrained(encoder_model_name_or_path)
    model.to(device)
    
    def evaluation_score_compute_metrics(pred):
        labels = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        preds = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        list_of_references = []
        for i in range(len(labels)):
            list_of_references.append([labels[i]])
        # calculate blue4
        blue4 = corpus_bleu(list_of_references=list_of_references, hypotheses=preds)
        return {"bleu4": blue4}

    # define trian/val
    dataset = load_dataset(args.dataset_huggingface, split="train")
    ds = dataset.train_test_split(test_size=0.1)
    train_ds = ds["train"]
    test_ds = ds["test"]

    train_dataset = ImageCaptioningDataset(dataset=train_ds, image_processor=feature_extractor, tokenizer=tokenizer)
    val_dataset = ImageCaptioningDataset(dataset=test_ds, image_processor=feature_extractor, tokenizer=tokenizer)

    # train arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        predict_with_generate=True,
        evaluation_strategy='steps',
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        overwrite_output_dir=True,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu4",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        num_train_epochs=args.num_train_epochs,
        save_total_limit=args.save_total_limit,
        generation_max_length=args.max_length,
        generation_num_beams=args.num_beams,
    )
    
    trainer = Seq2SeqTrainer(
            model=model,
            tokenizer=feature_extractor,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=evaluation_score_compute_metrics,
            data_collator=default_data_collator,
        )

    trainer.train()

    model.save_pretrained(args.output_dir)
    feature_extractor.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)