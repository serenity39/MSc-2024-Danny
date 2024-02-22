"""Script for pre-training BERT model from scratch.

This script loads the MSMARCO dataset, tokenizes the data, initializes
a BERT model for pre-training, and runs the training process.

  Typical usage example:

  python pretrain_bert.py
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    BertConfig,
    BertForPreTraining,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

# Configuration
PRETRAINING_DIR = "../data/collection/msmarco_passage/msmarco_v2_passage"
OUTPUT_DIR = "../data/results/"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "bert_pretrained.bin")
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 10000
MAX_STEPS = 100000

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MSMARCODataset(Dataset):
    """MSMARCO dataset for pre-training BERT."""

    def __init__(self, tokenizer, directory, max_len=512):
        """Inits MSMARCODataset with tokenizer, directory, and max sequence
        length."""
        self.tokenizer = tokenizer
        self.texts = []
        self.max_len = max_len

        # Load data
        for filename in os.listdir(directory):
            with open(
                os.path.join(directory, filename), "r", encoding="utf-8"
            ) as file:
                self.texts.append(file.read())

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.texts)

    def __getitem__(self, idx):
        """Returns the tokenized item at the given index."""
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
        }


def train(model, data_loader, optimizer, scheduler, device_, epoch, save_path):
    """Trains the BERT model.

    This function trains the model for one epoch using the provided data.
    It accumulates the loss and saves the model checkpoints.

    Args:
        model: The model to be trained.
        data_loader: The data loader for the training data.
        optimizer: The optimizer to use for training.
        scheduler: The scheduler to use for the learning rate decay.
        device: The device on which to train (e.g., 'cuda', 'cpu').
        epoch: The current epoch number (for checkpointing).
        save_path: The base path where to save model checkpoints.

    Returns:
        The total loss accumulated over the epoch.
    """
    model.train()
    total_loss = 0

    for step, batch in enumerate(data_loader):
        # Move each tensor in the batch to the specified device
        input_ids = batch["input_ids"].to(device_)
        attention_mask = batch["attention_mask"].to(device_)
        token_type_ids = batch["token_type_ids"].to(device_)

        # Reconstruct the batch on the device
        batch_on_device = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # Forward pass
        outputs = model(**batch_on_device)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Accumulate the loss
        total_loss += loss.item()

        # Logging after every 1000 steps
        if step % 1000 == 0:
            print(f"Step {step} - loss: {loss.item()}")

        # Checkpoint saving
        if step % 1000 == 0:
            checkpoint_path = (
                f"{save_path}_checkpoint_epoch_{epoch}_step_{step}.bin"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    return total_loss


def main():
    """Main function to run the pre-training."""
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load dataset
    dataset = MSMARCODataset(tokenizer, PRETRAINING_DIR, MAX_SEQ_LENGTH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize BERT model for pre-training
    config = BertConfig()
    model = BertForPreTraining(config)
    model.to(device)

    # Prepare optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # Run training
    for epoch in range(NUM_EPOCHS):
        total_loss = train(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device_=device,
            epoch=epoch,
            save_path=MODEL_SAVE_PATH,
        )
        print(f"Epoch {epoch} - Total loss: {total_loss}")

    # Save the pre-trained model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Pre-trained model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
