"""Fine-tune a BERT model on sequence classification task."""

import os

# Specify the GPU ID to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch  # noqa: E402
from datasets import DatasetDict, load_from_disk  # noqa: E402
from transformers import (  # noqa: E402
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

# Configs
DATASET_PATH = "../data/hf_datasets/depth_based_50_50"
MODEL_SAVE_PATH = "../data/results/models/depth_based_50_50/"
CHECKPOINT_PATH = "../data/results/checkpoints/depth_based_50_50/"


def tokenize_function(tokenizer, examples):
    """Tokenize the examples.

    Args:
        tokenizer: tokenizer to use.
        examples: examples to tokenize.
    """
    # Tokenize the query and document texts and return the inputs necessary
    return tokenizer(
        examples["query_text"],
        examples["doc_text"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )


# Ensure the GPU (if available) is used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset and tokenizer
dataset = load_from_disk(DATASET_PATH)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize the dataset
# Tokenize the dataset
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(tokenizer, examples), batched=True
)

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask", "relevance"],
)

# Split the dataset into training and validation sets
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
dataset_dict = DatasetDict(
    {
        "train": tokenized_dataset["train"],
        "validation": tokenized_dataset["test"],
    }
)

# Initialize the model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=CHECKPOINT_PATH,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../data/logs",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
)

trainer.train()

# Save the model
trainer.save_model(MODEL_SAVE_PATH)
