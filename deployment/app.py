"""
app.py
======
Gradio app for HF Spaces (ZeroGPU) — deploys RoBERTa-base (multiple-choice),
0.7544 MAP@3 on the Kaggle leaderboard.
Live demo: https://huggingface.co/spaces/Rohan014/smart-mcq-solver-roberta
"""

import spaces
import gradio as gr

from src.models.roberta import RoBERTaModel

_model = None


def get_model():
    global _model
    if _model is None:
        _model = RoBERTaModel().load()
    return _model


@spaces.GPU
def predict(prompt, opt_a, opt_b, opt_c, opt_d, opt_e):
    if not prompt or not prompt.strip():
        return "⚠️ Please enter a question.", {}

    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    if any(not (o or "").strip() for o in options):
        return "⚠️ Please fill in all 5 options (A-E).", {}

    model = get_model()
    proba = model.predict_proba_single(prompt, options)
    top3 = model.predict_top3_single(prompt, options)

    option_text = {"A": opt_a, "B": opt_b, "C": opt_c, "D": opt_d, "E": opt_e}
    medal = ["🥇", "🥈", "🥉"]
    answer_text = "\n".join(
        f"{medal[i]} **{letter}**: {option_text[letter]}" for i, letter in enumerate(top3)
    )

    confidences = {letter: float(score) for letter, score in zip("ABCDE", proba)}
    return answer_text, confidences


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Question", lines=3,
                    placeholder="e.g. What is the powerhouse of the cell?"),
        gr.Textbox(label="Option A"),
        gr.Textbox(label="Option B"),
        gr.Textbox(label="Option C"),
        gr.Textbox(label="Option D"),
        gr.Textbox(label="Option E"),
    ],
    outputs=[
        gr.Markdown(label="Top-3 Answer"),
        gr.Label(label="Confidence per option", num_top_classes=5),
    ],
    title="🧠 Smart MCQ Solver — RoBERTa-base",
    description=(
        "Answers 5-option multiple-choice questions using a fine-tuned "
        "**RoBERTa-base** model (AutoModelForMultipleChoice) — 0.7544 MAP@3 "
        "on the Kaggle leaderboard."
    ),
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
