from pathlib import Path
import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def create_heatmap(attention_matrix, input_tokens, output_tokens, results_dir, plot_color_scheme):
    """
    Creates a heatmap of the attention matrix, given the input and output tokens
    """

    fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(
        attention_matrix,
        xticklabels=input_tokens,
        yticklabels=output_tokens,
        cmap=plot_color_scheme,
        ax=ax
    )

    ax.set_xlabel("Input Tokens (Document)", fontsize=16)
    ax.set_ylabel("Output Tokens (Summary)", fontsize=16)
    ax.set_title("Token Level Cross-Attention", fontsize=18, pad=20)

    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(results_dir/"cross_attentions_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()


def filter_tokens(tokenizer, attention_matrix, inputs, outputs):
    """
    Takes attention matrix, input tokens, output tokens, and cleans them
    """

    #labels for the heat map
    #get the input token ids and use tokenizer to convert to text
    input_ids = inputs["input_ids"][0]
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    input_tokens_clean = [token.replace("Ġ", "") for token in input_tokens]

    #get the output token ids and use tokenizer to convert to text
    output_ids = outputs.sequences[0]
    output_tokens = tokenizer.convert_ids_to_tokens(output_ids)
    output_tokens_clean = [token.replace("Ġ", "") for token in output_tokens]

    #remove input tokens that are too influential, or punctuation
    tokens_to_remove = ['<s>', '</s>', '.', ';', 'Ċ', "\n", ""]

    filtered = []
    for i, token in enumerate(input_tokens_clean):
        if token not in tokens_to_remove:
            filtered.append(i)

    filtered_attention_matrix = attention_matrix[:, filtered]
    filtered_input_tokens = [input_tokens_clean[i] for i in filtered]

    #remove out tokens that are too influential, or punctuation
    filtered_output = []
    for i, token in enumerate(output_tokens_clean):
        if token not in tokens_to_remove:
            filtered_output.append(i)

    filtered_attention_matrix = filtered_attention_matrix[filtered_output, :]
    filtered_output_tokens = [output_tokens_clean[i] for i in filtered_output]

    return filtered_attention_matrix, filtered_input_tokens, filtered_output_tokens


def create_attention_matrix(cross_attentions):
    """
    Create the attention matrix from the cross attentions
    """

    #for each output token, average the attention across all heads in the last layer
    attention_matrix = []
    for cross_attention in cross_attentions:
        cross_attention = cross_attention[-1]
        cross_attention = cross_attention.squeeze()
        attention_score = cross_attention.mean(dim=(0, 1)) 
        attention_matrix.append(attention_score.numpy())

    #matrix of output token row, input token columns
    attention_matrix = np.array(attention_matrix)

    return attention_matrix


def main():

    #load config file
    config_file = Path("config/config_model40k.yaml").read_text()
    config = yaml.safe_load(config_file)

    #set variables
    model_name = config["model"]["model_name"]
    input_document = config["cross_attentions_heatmap"]["input_document"]
    summary_size = config["cross_attentions_heatmap"]["summary_size"]
    plot_color_scheme = config["cross_attentions_heatmap"]["plot_color_scheme"]
    results_dir = Path(config["cross_attentions_heatmap"]["results_dir"])

    #load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, attn_implementation="eager")

    model.eval()

    #tokenize input text
    inputs = tokenizer(input_document, return_tensors="pt", truncation=True, max_length=1024)

    #find cross attentions and outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            output_attentions=True,
            return_dict_in_generate=True,
            max_new_tokens=summary_size
        )

    cross_attentions = outputs.cross_attentions

    attention_matrix = create_attention_matrix(cross_attentions)

    filtered_attention_matrix, filtered_input_tokens, filtered_output_tokens = filter_tokens(tokenizer, attention_matrix, inputs, outputs)

    create_heatmap(filtered_attention_matrix, filtered_input_tokens, filtered_output_tokens, results_dir, plot_color_scheme)


if __name__ == "__main__":
    main()