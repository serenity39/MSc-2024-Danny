"""Script for pre-training BERT model from scratch.

This script loads the MSMARCO dataset, tokenizes the data, initializes
a BERT model for pre-training, and runs the training process.

  Typical usage example:

  python pretrain_bert.py
"""

import csv
import logging
import os

# Specify the GPU ID to use
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score, ndcg_score  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import logging as tf_logging  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
tf_logging.set_verbosity_error()

# Configuration

# Path to the data
INPUT_TEXT = "../data/trainingsets/inputs_depth_50_50.csv"
OUTPUT_DIR = "../data/results/depth_50_50/"
VAL_DATA = "../data/validationdata/validation_data.csv"
# Change name of model
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "bert_depth_50_50.pt")
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
WARMUP_STEPS = 10000
MAX_STEPS = 100000

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    """Dataset for training BERT."""

    def __init__(self, tokenizer, input_file, max_len=512):
        """Inits the dataset class."""
        self.tokenizer = tokenizer
        self.data = []
        self.max_len = max_len

        # Read the CSV data file
        with open(input_file, "r", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                self.data.append((row[0], row[1], int(row[2])))

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns the tokenized item at the given index."""
        # Separate the query and passage based on the special token
        query, passage, label = self.data[idx]
        encoding = self.tokenizer.encode_plus(
            query,
            passage,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        # Return the tokenized input and the label
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def train(model, data_loader, optimizer, scheduler, device_, epoch, save_path):
    """Trains the BERT model.

    This function trains the model for one epoch using the provided data.
    It returns the average loss over the epoch and saves the model checkpoints.

    Args:
        model: The model to be trained.
        data_loader: The data loader for the training data.
        optimizer: The optimizer to use for training.
        scheduler: The scheduler to use for the learning rate decay.
        device: The device on which to train (e.g., 'cuda', 'cpu').
        epoch: The current epoch number (for checkpointing).
        save_path: The base path where to save model checkpoints.

    Returns:
        float: The average loss over the epoch.
    """
    model.train()
    total_loss = 0

    total_steps_per_epoch = len(data_loader)
    checkpoint_interval = total_steps_per_epoch // 10

    for step, batch in enumerate(data_loader):
        # Move each tensor in the batch to the specified device
        input_ids = batch["input_ids"].to(device_)
        attention_mask = batch["attention_mask"].to(device_)
        token_type_ids = batch["token_type_ids"].to(device_)
        labels = batch["labels"].to(device_)

        # Reconstruct the batch on the device
        batch_on_device = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

        # Forward pass
        outputs = model(**batch_on_device)
        loss = outputs.loss

        if loss is None:
            logging.error(f"Loss is None at step {step}.")
            continue  # Skip the rest of the loop and move to the next batch

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate the loss
        total_loss += loss.item()

        # Logging loss
        if step % 100 == 0:
            logging.info(f"Step {step} - loss: {loss.item()}")

        # Checkpoint saving
        if step % checkpoint_interval == 0 and step > 0:
            checkpoint_path = (
                f"{save_path}_checkpoint_epoch_{epoch}_step_{step}.pt"
            )
            checkpoint = {
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    avg_loss = total_loss / total_steps_per_epoch
    return avg_loss


import numpy as np
import torch
from sklearn.metrics import average_precision_score, ndcg_score


def evaluate_model(model, val_data_loader, device_):
    """Evaluates the BERT model on the validation data.

    This function evaluates the model on the validation data and returns the
    MAP and NDCG metrics.

    Args:
        model: The model to be evaluated.
        val_data_loader: The data loader for the validation data.
        device_: The device on which to evaluate (e.g., 'cuda', 'cpu').

    Returns:
        float: The mean average precision (MAP) score.
        float: The normalized discounted cumulative gain (NDCG) score.
    """
    model.eval()
    total_map = 0.0
    total_ndcg = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_data_loader:
            inputs = {
                "input_ids": batch["input_ids"].to(device_),
                "attention_mask": batch["attention_mask"].to(device_),
                "token_type_ids": batch.get(
                    "token_type_ids", torch.tensor([])
                ).to(device_),
            }
            labels = batch["labels"].to(device_)  # Binary relevance scores
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = (
                torch.sigmoid(logits)[:, 1].cpu().numpy()
            )  # Assuming column 1 is the positive class

            # Ensure labels is a binary array for each sample
            labels_np = labels.cpu().numpy()

            # Calculate MAP and NDCG for the batch
            batch_map = average_precision_score(labels_np, probabilities)
            batch_ndcg = ndcg_score([labels_np], [probabilities])

            total_map += batch_map
            total_ndcg += batch_ndcg
            num_batches += 1

    # Calculate average scores over all batches
    avg_map = total_map / num_batches
    avg_ndcg = total_ndcg / num_batches

    return avg_map, avg_ndcg


def main():
    """Main function to run the pre-training."""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load dataset
    dataset = CustomDataset(tokenizer, INPUT_TEXT, MAX_SEQ_LENGTH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = CustomDataset(tokenizer, VAL_DATA, MAX_SEQ_LENGTH)
    val_data_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize BERT model for fine-tuning
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )
    model.to(device)

    # Prepare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # Run training
    for epoch in range(NUM_EPOCHS):
        avg_loss = train(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device_=device,
            epoch=epoch,
            save_path=MODEL_SAVE_PATH,
        )
        logging.info(f"Epoch {epoch} - Average loss: {avg_loss}")

    map_score, ndcg_value = evaluate_model(model, val_data_loader, device)
    logging.info(
        f"Validation - Epoch: {epoch}, MAP: {map_score}, NDCG: {ndcg_value}"
    )

    # Save the pre-trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"Pre-trained model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
