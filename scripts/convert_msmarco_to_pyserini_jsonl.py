"""Convert the MS MARCO dataset to the JSONL format required by Pyserini."""

import json
import os


def transform_file(input_file, output_file):
    """Convert a file from MS MARCO format to the format required by Pyserini.

    Convert a single file from the MS MARCO format to the JSONL format
    required by Pyserini. The format of the input file is as follows:
    {
        "id": "docid",
        "contents": "document text"
    }

    Args:
        input_file (str): The file path for the input JSONL file.
        output_file (str): The file path for the output JSONL file.

    Returns:
        None: The function writes the converted file to 'output_file'.
    """
    # Ensure the output file has the .jsonl extension
    output_file = f"{output_file}.jsonl"

    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            data = json.loads(line)
            transformed_data = {
                "id": data["docid"],
                "contents": data["passage"],
            }
            outfile.write(json.dumps(transformed_data) + "\n")


def transform_directory(input_dir, output_dir):
    """Convert files in input dir and writes files to the output dir.

    Loop through the input directory and convert all files in the directory by
    using the transform_file function.

    Args:
        input_dir (str): The directory containing input JSONL files.
        output_dir (str): The directory where the output JSONL files
            will be written.

    Returns:
        None: The function writes the converted files to 'output_dir'.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_files = 0

    # Iterate over all files in the input directory
    for input_file in os.listdir(input_dir):
        input_file_path = os.path.join(input_dir, input_file)
        # Check if it's a file and not a subdirectory
        if os.path.isfile(input_file_path):
            output_file_path = os.path.join(output_dir, input_file)
            transform_file(input_file_path, output_file_path)
            num_files += 1
            print(
                f"Converted and wrote: {output_file_path}. "
                f"Converted {num_files}/{len(os.listdir(input_dir))} files."
            )


if __name__ == "__main__":
    input_dir_path = (
        "/home/serenity39/master-thesis/data/collections/msmarco-passage/"
        "msmarco_v2_passage"
    )
    output_dir_path = (
        "/home/serenity39/master-thesis/data/collections/msmarco-passage/"
        "pysernini-format"
    )

    transform_directory(input_dir_path, output_dir_path)
