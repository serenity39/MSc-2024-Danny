# Preprocessing MSMARCO Dataset for BERT Pre-Training

## Overview

This guide outlines the process for preprocessing the MSMARCO dataset for use in BERT pre-training.

## Download the Dataset

Go to the [MSMARCO dataset page](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html) and download:

  - msmarco_v2_passage.tar
  - passv2_dev_queries.tsv
  - passv2_dev_top100.txt.gz

This can be done with the command (change `name-of-file` to the name of the dataset file you want to download):

```bash
wget --header "X-Ms-Version: 2019-12-12" https://msmarco.z22.web.core.windows.net/msmarcoranking/name-of-file
```

## Organize the data

To extract a tar archive containing gzipped JSONL files into a folder where each gzipped file is decompressed into its own .jsonl file, you can use the following command-line instructions.
First, ensure you have `tar` and `gzip` available on your system.
Most Unix-like operating systems, including Linux and macOS, have these tools by default.
On Windows, you can use tools like `7-zip` instead.

Follow these steps to extract and decompress the files:

1. Extract the tar archive into a directory:

```bash
tar -xvf msmarco_v2_passage.tar -C path/to/extract/jsonl_files
```

2. Navigate to the directory where you've extracted the gzipped files and decompress them. You can decompress all gzipped files in the directory by using:

```bash
gzip -d *.gz
```

After following these steps, you should have a folder filled with uncompressed .jsonl files, ready for processing.
You might have to make sure that the decompressed files have the `.jsonl` extension at the end.
If not, you can do this by using the command in the directory with the files:

```bash
for file in *; do mv "$file" "$file.jsonl"; done
```

## Preprocess the data

You can run the `data_prep.py` script to pre-process the data by navigating to the `/scripts` folder and running the command:

```bash
python data_prep.py
```

This script will:

  - Load the queries and passage IDs
  - Extract the passage text
  - Merge queries with corresponding passages
  - Output the preprocessed data for BERT pre-training

## Verify the Output

Check the `data/msmarco/preprocessed` directory for the output file.
The output will be a text file with each line formatted as `query [SEP] passage`, suitable for BERT pre-training.

## Additional Notes

Ensure that the paths in the `data_prep.py` script matches the paths to your collections/datasets.
