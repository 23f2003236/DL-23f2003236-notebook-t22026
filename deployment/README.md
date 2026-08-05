# Smart MCQ Solver (Deployment)

5-option multiple-choice question answering, deployed on Hugging Face Spaces
using Gradio + ZeroGPU. Deployed model: RoBERTa-base
(0.7544 MAP@3 on the Kaggle leaderboard, tied for best individual model
in this project).

## Live Demo

👉 [Try it here](https://huggingface.co/spaces/Rohan014/smart-mcq-solver-roberta)

## How it works

The demo loads the fine-tuned RoBERTa-base model from `src/models/roberta.py`,
takes a question + 5 options as input, and returns the top-3 most likely
answers with a confidence score per option.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Requires `outputs/checkpoints/roberta_best.pt` and the `src/` package from
the project root.
