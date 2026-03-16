from pathlib import Path
import yaml
import pandas as pd
import nltk
import subprocess
import sys

def create_train_csv(train_data, data_directory):
    """
    Create training data
    """
    test_data = pd.concat(train_data, ignore_index=True)
    test_data.to_csv(data_directory/"train_notclean_40k.csv", index=False)
    #had to add this or the data didn't work
    df = pd.read_csv(data_directory/"train_notclean_40k.csv", on_bad_lines='skip')
    df.to_csv(data_directory/"train_40k.csv", index=False)
    df.head()


def create_val_csv(val_data, data_directory):
    """
    Create validation data
    """
    df = pd.concat(val_data, ignore_index=True)
    df.to_csv(data_directory/"val_40k.csv", index=False)
    df.head()


def main():  
    nltk.download('punkt_tab')

    #load config file
    config_file = Path("config/config_model40k.yaml").read_text()
    config = yaml.safe_load(config_file)

    #set variables
    data_directory = Path(config["dataset"]["data_dir"])
    base_model_name = config["train_args"]["base_model_name"]
    output_dir = Path(config["train_args"]["output_dir"])
    train_batch_size = config["train_args"]["train_batch_size"]
    eval_batch_size = config["train_args"]["eval_batch_size"]
    document_column = config["train_args"]["document_column"]
    summary_column = config["train_args"]["summary_column"]
    num_workers = config["train_args"]["num_workers"]

    #read in datasets
    train_data = []
    for url in config["dataset"]["train"]:
        train_data.append(pd.read_parquet(url))

    val_data = []
    for url in config["dataset"]["validation"]:
        val_data.append(pd.read_parquet(url))

    create_train_csv(train_data, data_directory)

    create_val_csv(val_data, data_directory)

    try:
        result = subprocess.run([sys.executable,
            "run_summarization.py",
            "--model_name_or_path", f"{base_model_name}",
            "--do_train",
            "--do_eval",
            "--train_file", f"{data_directory}/train_40k.csv",
            "--validation_file", f"{data_directory}/val_40k.csv",
            "--output_dir", f"{output_dir}",
            f"--per_device_train_batch_size={train_batch_size}",
            f"--per_device_eval_batch_size={eval_batch_size}",
            "--predict_with_generate",
            "--text_column", f"{document_column}",
            "--summary_column", f"{summary_column}",
            f"--preprocessing_num_workers={num_workers}"
        ])
    except KeyboardInterrupt:
        print("Exiting...")


if __name__ == "__main__":
    main()