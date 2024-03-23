"""Fine-tune a BERT model on sequence classification task."""

import logging
import os

# Specify the GPU ID to use
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch  # noqa: E402
from datasets import DatasetDict, load_from_disk  # noqa: E402
from transformers import (  # noqa: E402
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configs
DATASET_PATH = "../data/hf_datasets/test/shallow_based_5000_2"
MODEL_SAVE_PATH = "../data/results/models/early_stopping/shallow_based_5000_2"
CHECKPOINT_PATH = (
    "../data/results/checkpoints/early_stopping/shallow_based_5000_2"
)


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
logging.info("Tokenizing the dataset...")
tokenized_dataset = dataset.map(
    lambda examples: tokenize_function(tokenizer, examples), batched=True
)

# Rename the relevance column to labels
tokenized_dataset = tokenized_dataset.map(lambda e: {"labels": e["relevance"]})

tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
)

# To check tokenization
for x in tokenized_dataset["input_ids"][:2]:
    print(tokenizer.decode(x))

# Split the dataset into training and validation sets
logging.info("Splitting the dataset into training and validation sets...")
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)
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
    save_strategy="epoch",
    num_train_epochs=10,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../data/logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3, early_stopping_threshold=0.001
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    callbacks=[early_stopping],
)

logging.info("Training the model...")
model.to(device)
trainer.train()

# Save the model
logging.info("Saving the model and tokenizer...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
