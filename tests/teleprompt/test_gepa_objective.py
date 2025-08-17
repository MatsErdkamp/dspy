# If needed:
# pip install dspy-ai datasets

# =========================
# 0) LMs and global config
# =========================


import dspy
from dspy.teleprompt.gepa.gepa import GEPA
from dspy.utils.callback import BaseCallback


PROGRAM_LM = "ollama/gemma3:1b"  # same as tutorial default
REFLECT_LM = "ollama/gemma3:1b"  # same as tutorial snippet

lm = dspy.LM(PROGRAM_LM, temperature=1, max_tokens=32000)
dspy.configure(lm=lm)

dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


# 1. Define a custom callback class that extends BaseCallback class
class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run a custom logging code.
    def on_module_end(self, call_id, outputs, exception):
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())


# 3. Set the callback to DSPy setting so it will be applied to program execution
dspy.configure(callbacks=[AgentLoggingCallback()])

# =========================
# 1) Load AIME datasets
# =========================
from datasets import load_dataset
import random


def init_dataset(max_examples=10):
    # Train/val from prior AIME years (with written solutions for feedback)
    train_split = load_dataset("AI-MO/aimo-validation-aime")["train"]
    train_split = [
        dspy.Example(
            {
                "problem": x["problem"],
                "solution": x["solution"],
                "answer": x["answer"],
            }
        ).with_inputs("problem")
        for x in train_split
    ]
    random.Random(0).shuffle(train_split)
    # Limit to max_examples
    train_split = train_split[:max_examples]
    tot = len(train_split)

    # Test from AIME 2025 (no solutions provided), repeated 5x for stability
    test_split = load_dataset("MathArena/aime_2025")["train"]
    test_split = [
        dspy.Example(
            {
                "problem": x["problem"],
                "answer": x["answer"],
            }
        ).with_inputs("problem")
        for x in test_split
    ]
    # Limit test split to max_examples as well
    test_split = test_split[:max_examples]

    train_set = train_split[: tot // 2]
    val_set = train_split[tot // 2 :]
    test_set = test_split * 5
    return train_set, val_set, test_set


train_set, val_set, test_set = init_dataset(max_examples=20)


# =========================
# 2) Program: CoT for AIME
#    (add 'reasoning' to signature)
# =========================
class GenerateResponse(dspy.Signature):
    """Solve the problem with concise reasoning and return the final integer answer."""

    problem = dspy.InputField()
    reasoning = dspy.OutputField()
    answer = dspy.OutputField()


program = dspy.Predict(GenerateResponse)


# ==========================================
# 3) Subscore helpers (pluggable if you want)
# ==========================================
def parse_int_answer(x):
    try:
        return int(str(x).strip())
    except Exception:
        return None


def answer_correctness(gold_answer, pred_answer) -> float:
    """1.0 for exact integer match, else 0.0 (AIME answers are integers)."""
    g = parse_int_answer(gold_answer)
    p = parse_int_answer(pred_answer)
    return 1.0 if (g is not None and p is not None and g == p) else 0.0


def reasoning_strength(problem: str, reasoning: str, pred_answer, gold_answer) -> float:
    """
    Lightweight rubric in [0,1] that rewards structured, concise math reasoning.
    Heuristic signals (you can replace with your own judge):
      - Non-empty, not wildly long (token-length taper).
      - Contains math-y cues (equations, 'mod', variables).
      - Mentions the final numeric answer (sanity).
    """
    if not reasoning:
        return 0.0
    text = str(reasoning)
    toks = text.split()
    # Length score: good if ~20-200 tokens, taper otherwise
    if len(toks) <= 5:
        len_score = 0.1
    elif len(toks) <= 200:
        len_score = 1.0
    elif len(toks) <= 400:
        len_score = 0.6
    else:
        len_score = 0.3

    has_equation = any(sym in text for sym in ["=", "≡", "+", "-", "×", "*", "/", "^"])
    has_math_cue = any(
        w in text.lower() for w in ["mod", "let", "assume", "therefore", "hence"]
    )
    math_score = 1.0 if (has_equation or has_math_cue) else 0.5

    pa = parse_int_answer(pred_answer)
    mentions_pred = (pa is not None) and (str(pa) in text)
    # Don't require gold mention (that would leak labels), only predicted.
    anchor_score = 1.0 if mentions_pred else 0.7

    # Mild penalty if predicted answer is unparsable
    parse_penalty = 0.8 if parse_int_answer(pred_answer) is None else 1.0

    r = max(0.0, min(1.0, 0.4 * len_score + 0.4 * math_score + 0.2 * anchor_score))
    return max(0.0, min(1.0, r * parse_penalty))


# ==========================================
# 4) Metrics that return subscores + feedback
#    (use one for plain eval; one with richer feedback for GEPA)
# ==========================================
def metric_with_subscores(
    example, prediction, trace=None, pred_name=None, pred_trace=None
):
    gold = example["answer"]
    prob = example["problem"]
    pred_ans = getattr(prediction, "answer", None)
    pred_cot = getattr(prediction, "reasoning", "")

    a = answer_correctness(gold, pred_ans)
    r = reasoning_strength(prob, pred_cot, pred_ans, gold)
    # Weighted top-level — tweak as you like
    total = 0.8 * a + 0.2 * r

    return dspy.Prediction(
        score=total,
        subscores={"answer": a, "reasoning": r},
        feedback=f"Answer={a:.2f}, Reasoning={r:.2f}, Total={total:.2f}",
    )


def metric_with_feedback_and_subscores(
    example, prediction, trace=None, pred_name=None, pred_trace=None
):
    gold = example["answer"]
    prob = example["problem"]
    soln = example.get("solution", "")
    pred_ans = getattr(prediction, "answer", None)
    pred_cot = getattr(prediction, "reasoning", "")

    a = answer_correctness(gold, pred_ans)
    r = reasoning_strength(prob, pred_cot, pred_ans, gold)
    total = 0.8 * a + 0.2 * r

    print(f"Score: {total}")

    # Human-readable coaching for GEPA reflection LM
    g_int = parse_int_answer(gold)
    p_int = parse_int_answer(pred_ans)
    if p_int is None:
        fb = (
            f"Final answer must be a valid integer (AIME). "
            f"You returned '{pred_ans}'. Ensure the 'answer' is only an integer with no extra text. "
        )
    else:
        fb = (
            "Your answer is correct. "
            if a == 1.0
            else f"Your answer is incorrect. Correct answer is '{g_int}'. "
        )

    if soln:
        fb += (
            "Here's an official step-by-step solution. Reflect on what to take away, "
            "and adjust your approach/prompt accordingly:\n" + soln
        )

    # Add short rubric pointers for reasoning quality
    fb += (
        "\nReasoning rubric reminders: be concise (20–200 tokens), include concrete steps/equations, "
        "and explicitly anchor the final numeric answer."
    )

    return dspy.Prediction(
        score=total, subscores={"answer": a, "reasoning": r}, feedback=fb
    )


# ==========================================
# 5) Baseline evaluation (unoptimized CoT)
# ==========================================
# evaluate = dspy.Evaluate(
#     devset=test_set,
#     metric=metric_with_subscores,  # uses subscores even for baseline eval
#     display_table=True,
#     display_progress=True,
# )
# evaluate(program)

# ==========================================
# 6) GEPA optimization with subscores-aware metric
# ==========================================
from dspy import MIPROv2

optimizer = GEPA(
    metric=metric_with_feedback_and_subscores,  # subscores + rich feedback
    auto="light",
    num_threads=32,
    reflection_minibatch_size=3,
    reflection_lm=dspy.LM(model=REFLECT_LM),
    pareto_frontier_type="hybrid",
)


optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)

# (Optional) Inspect the evolved instructions
try:
    print(optimized_program.signature.instructions)
except Exception:
    pass

# ==========================================
# 7) Re-evaluate the optimized program
# ==========================================
evaluate(optimized_program)
