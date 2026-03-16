import torch

def get_token_score_pairs(article, model, tokenizer, device):
    """
    This function takes an article and breaks down each token to give a list of tuples (token, score)
    This only works if we use a sectioned article with \n as we will use that to detect new lines

    Parameters
    ----------
    article: article section (str)
        article that includes a \n a seperator which will be used for aggregation next step
    
    model: model 
       model used for scoring

    tokenizer: tokenizer
        tokenizer used for scoring

    device: device
        device, prioritze cuda otherwise cpu is fine

    Returns
    -------
    list of tuples (token, score)
        for each token there would be an overall score associated with the token
    """

    inputs = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=1024
    ).to(device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True
        )

    cross_attentions = outputs.cross_attentions
    last_layer = cross_attentions[-1] #get the last layer, most semantic value
    avg_heads = last_layer.mean(dim=1) #average over all heads 
    token_importance = avg_heads.mean(dim=1) #average over all tokens

    input_ids = inputs["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    token_scores = [(token, float(score.item())) for token, score in zip(tokens, token_importance[0])]

    return token_scores


def rank_important_lines(token_scores, average_score = False):
    """"
    This function ranks lines based on their importance divided by the amount of tokens
    This only works if we use a sectioned article with \n as we will use that to detect new lines

    Parameters
    ----------
    token_scores: list of tuples (token, score)
        token_scores is a list of tuples where each tuple contains a token and its corresponding importance score
    
    average_score: bool (default False)
        if True then it will rank based on the average score of all tokens in a line
        if False then it will rank based on the sum of all tokens in a line

    Returns
    -------
    list of tuples (line, total_score, avg_score, line position)
        this function returns a list of tuples for token line, overall token score, average score across tokens, and where the line is in the article
    """
    lines = []
    current_line = []
    current_score = 0
    line_count = 0

    newline_count = sum(1 for token, _ in token_scores if token == "Ċ")
    total_lines = newline_count + 1

    for token, score in token_scores:
        if token != "Ċ": #this token symbolizes new line so if it isn't a new line keeping adding tokens
            current_line.append(token)
            current_score += score
        else:
            if current_line != []: #check if current line is empty or not
                avg_score = current_score / len(current_line) #calculate average score for the line
                lines.append((" ".join(current_line).replace("Ġ","").strip(' . '), current_score, avg_score, line_count / total_lines)) 
            current_line = []
            current_score = 0
            line_count += 1

    if current_line != []:
        avg_score = current_score / len(current_line) #calculate average score for the line
        lines.append((" ".join(current_line).replace("Ġ","").strip(' . '), current_score, avg_score, line_count / total_lines))

    lines[0]  = (lines[0][0].replace("<s>", "").strip(),  lines[0][1], lines[0][2], lines[0][3]) #take out starting token
    lines[-1] = (lines[-1][0].replace("</s>", "").strip(), lines[-1][1], lines[-1][2], lines[-1][3]) #take out ending token

    if average_score:
        ranked_lines = sorted(lines, key=lambda x: x[2], reverse=True) #ranks the lines based on the average score in desc order
    else:
        ranked_lines = sorted(lines, key=lambda x: x[1], reverse=True) #ranks the lines based on the total score in desc order

    return ranked_lines


def rank_lines(article, model, tokenizer, device, average_score = False):

    get_token_score_pairs(article, model, tokenizer, device)
    ranked_lines = rank_important_lines(get_token_score_pairs, average_score)

    return ranked_lines