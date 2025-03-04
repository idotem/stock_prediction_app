import os
import subprocess

import markdown
import numpy as np
import torch
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import colbert_score

checkpoint = 'colbert-ir/colbertv2.0'
config = ColBERTConfig(doc_maxlen=300, nbits=2)

ckpt = Checkpoint(checkpoint, colbert_config=config)


def convert_markdown_to_html(markdown_text):
    return markdown.markdown(markdown_text)


def rank_with_colbert(query, answers):
    """Use ColBERT to rerank the retrieved GraphRAG results."""
    Q = ckpt.queryFromText([query])
    print(f"Q: {Q}")
    D = ckpt.docFromText(answers, bsize=32)[0]
    print(f"D: {D}")
    D_mask = torch.ones(D.shape[:2], dtype=torch.long)
    print(f"D_mask: {D_mask}")
    scores = colbert_score(Q, D, D_mask).flatten().cpu().numpy().tolist()
    print(f"scores: {scores}")
    ranking = np.argsort(scores)[::-1]
    print(f"ranking: {ranking}")
    return answers[ranking[0]]


def answer_question(question):
    try:
        os.environ["PATH"] += ":/home/meto/.local/bin"  # Adjust this path accordingly

        print(
            f"Executing local search: /home/meto/personal-projects/stock_prediction_app/query-graphrag.sh '{question}'")
        answer_local = query_graphrag(question, "local")

        print(
            f"Executing global search: /home/meto/personal-projects/stock_prediction_app/query-graphrag.sh '{question}'")
        answer_global = query_graphrag(question, "global")

        answer_local = answer_local.rpartition('SUCCESS:')[2]
        answer_global = answer_global.rpartition('SUCCESS:')[2]
        answers = [ans for ans in [answer_local, answer_global] if ans]
        print(f"Answers: {answers}")

        if not answers:
            return "No valid response from GraphRAG."

        best_answer = rank_with_colbert(question, answers)
        print(f"best_answer: {best_answer}")
        processed_text = convert_markdown_to_html(best_answer)
        print(f"processed_text{processed_text}")
        return processed_text
    except subprocess.CalledProcessError as e:
        print("Error: ", e)
        return f"An error aragraph occurred: {e}"


def query_graphrag(question, context):
    result = subprocess.run(['/home/meto/personal-projects/stock_prediction_app/query-graphrag.sh',
                             question, context],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                            )
    err = result.stderr
    if err:
        print(err)
    return result.stdout
