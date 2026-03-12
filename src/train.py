from datasets import Dataset
from pathlib import Path
from transformers import BartTokenizer, Seq2SeqTrainer,DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, BartForConditionalGeneration
import torch

from src.load_and_clean import load_and_clean

def tokenize_document(dataset, model_name):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model_input = tokenizer(dataset["article"], max_length = 1024, truncation = True)
    labels = tokenizer(dataset["abstract"], max_length = 320, truncation = True)

    model_input["labels"] = labels['input_ids']
    return model_input


def train(train, validation, device, model_name, args):
    """
    Trains the model on the dataset
    """

    train_dataset = Dataset(load_and_clean(train))
    validation_dataset = Dataset(load_and_clean(validation))

    tokenized_dataset = train_dataset.map(tokenize_document, batched=True)
    validation_tokenized = validation_dataset.map(tokenize_document, batched=True, remove_columns=validation_dataset.column_names)

    model = BartForConditionalGeneration.from_pretrained(model_name, torch_dtype = torch.bfloat16).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding="longest"  
        )
    
    training_args = Seq2SeqTrainingArguments(
        **args["training"]
        )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=validation_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator
        )

    trainer.train()

if __name__ == "__main__":
    train()