# Smart MCQ Solver

A machine learning project that answers multiple-choice questions (5 options: A, B, C, D, E).
Made for a Deep Learning course project, based on a Kaggle-style competition called
**Smart MCQ Solver Challenge**.

## What this project does

Given a question and 5 options, the model predicts the top-3 most likely correct answers.
Scoring is done using **MAP@3** (Mean Average Precision at 3).

This project uses **4 different models** and combines them together:

| Model | Type | Leaderboard Score (MAP@3) |
|---|---|---|
| TF-IDF + Logistic Regression | Classical ML baseline | 0.7511 |
| BiLSTM | Deep learning, trained from scratch (no pretrained weights) | 0.7540 |
| DeBERTa-v3-small | Pretrained transformer, fine-tuned | 0.7544 |
| RoBERTa-base | Pretrained transformer, fine-tuned (multiple-choice head) | 0.7544 |
| **Final Ensemble (all 4 combined)** | Weighted rank voting | **0.7602** |

The final ensemble score (0.7602) is better than every single model alone.

## Project structure

```
DL-23f2003236-notebook-t22026/
├── data/ # train.csv, test.csv, sample_submission.csv
├── notebooks/ # Original working notebooks (EDA, baseline, LSTM, DeBERTa, RoBERTa, ensemble)
│ └── Smart_MCQ_Solver_Final.ipynb # Everything combined in one notebook
├── milestones/ # Milestone work saved during the project
├── deployment/ # Standalone Gradio app deployed to HF Spaces
│ ├── app.py
│ ├── requirements.txt
│ └── README.md # Live demo link + deployment notes
├── src/ # Clean, reusable production code
│ ├── config.py # All hyperparameters and paths
│ ├── preprocessing.py # Text cleaning + prompt building
│ ├── dataset.py # PyTorch Dataset classes
│ ├── metrics.py # MAP@3 metric
│ ├── ensemble.py # Weighted rank ensemble logic
│ ├── inference.py # Loads models, runs predictions
│ ├── predict.py # Main function used by the app
│ └── models/ # Model architectures (tfidf, lstm, deberta, roberta)
├── kaggle_notebook.ipynb # Notebook used for Kaggle submission
├── project_report.pdf # Full project report
├── requirements.txt
└── .gitignore
```

## How the models work

1. **TF-IDF + Logistic Regression** — Turns each question + options into one text
   and classifies it with a simple ML model. Fast baseline.

2. **BiLSTM (from scratch)** — A word tokenizer and a 2-layer bidirectional LSTM,
   built and trained completely from scratch. No pretrained model used here.

3. **DeBERTa-v3-small** — Each (question, option) pair is scored one at a time as
   "correct" or "wrong". Uses a pretrained transformer backbone with a custom
   classification head.

4. **RoBERTa-base** — Uses HuggingFace's built-in multiple-choice model. All 5
   options for a question go through the model together in one pass.

5. **Ensemble** — Each model gives its own top-3 answers. All 4 model's answers are
   combined using weighted voting (weights: TF-IDF 0.15, LSTM 0.20, DeBERTa 0.40,
   RoBERTa 0.25) to get the final top-3 prediction.

## kaggle lb score 
<img width="1444" height="98" alt="image" src="https://github.com/user-attachments/assets/3eec9e79-70c3-4415-97ea-0ac4fe8b648b" />


## Important note: data leakage found

During EDA, 242 duplicate question rows were found inside `train.csv` (some
questions appear twice with different IDs), and 267 questions are shared between
`train.csv` and `test.csv`. This means local validation scores (like accuracy on
the val set) look very high (close to 1.0) — but this is because of memorized
duplicate rows, not because the model is actually that good.

**The real, trustworthy score is the Kaggle leaderboard score** (shown in the table
above), because the leaderboard test answers were never seen during training.

## Experiment tracking

All 3 deep learning models (LSTM, DeBERTa, RoBERTa) log their training progress to
**Weights & Biases (WandB)** — training/validation loss, MAP@3, accuracy, and F1
score, every epoch. This makes it easy to compare all 3 runs side by side.

## How to run

1. Clone the repository:
```bash
   git clone https://github.com/23f2003236/DL-23f2003236-notebook-t22026.git
   cd DL-23f2003236-notebook-t22026
```
2. Install requirements:
```bash
   pip install -r requirements.txt
```
3. Open `notebooks/final_nb.ipynb` and run all cells top to bottom.
4. Make sure `data/train.csv` and `data/test.csv` are present (or update the path
   to the Kaggle dataset location).

## Full report

See `project_report.pdf` for the complete write-up — methodology, training details,
error analysis, and conclusions.

## Author

Rohan Kumar (23f2003236)
Course: BSDA2001P — Deep Learning and Generative AI
