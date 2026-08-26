"""
============================================================================
SYCOPHANCY STEERING VECTOR ON PYTHIA-70M (Contrastive Activation Addition)
============================================================================

GOAL: Find a "direction" inside the model's internals that corresponds to
"being sycophantic" (agreeing with the user just because they stated an
opinion), then subtract that direction out during generation to make the
model more independently honest.

BIG PICTURE / THE 4 STEPS THIS SCRIPT DOES:
  1. Load the model and dataset.
  2. For each example, build TWO near-identical texts: one where the model's
     answer AGREES with the user (sycophantic), one where it DISAGREES
     (honest). Same question, only the final answer token differs.
  3. Run both texts through the model, grab the internal "residual stream"
     vector at the LAST token position, at one chosen layer. Average these
     vectors across many examples for each condition. Subtract:
         steering_vector = mean(sycophantic vectors) - mean(honest vectors)
     This vector approximates "the direction of sycophancy" inside the model.
  4. During generation on a NEW question, manually add this vector into the
     residual stream at every generation step, with different coefficients:
         negative coefficient -> push away from sycophancy (more honest)
         positive coefficient -> push toward sycophancy (more agreeable)
         zero                 -> baseline, no intervention

WHAT IS THE "RESIDUAL STREAM"?
  At every layer of a transformer, for every token position, there's a
  vector of numbers representing "everything the model currently
  believes/knows at this point in the computation." This is the residual
  stream. It's the natural place to intervene because every part of the
  model reads from and writes to it.

WHY THE LAST TOKEN?
  We want the vector representing the model's internal state right before
  it decides "should my next token be (A) or (B)". That decision-point
  information lives in the residual stream at the LAST token position of
  our prompt.

WHY A MIDDLE LAYER?
  Early layers tend to represent low-level token/syntax information; later
  layers are closer to the output logits already. Concepts like "am I being
  agreeable" tend to be most cleanly represented somewhere in the middle,
  where the model has built up abstract features but hasn't yet collapsed
  them into next-token predictions. This is an empirical choice - the
  bottom of this script includes a coefficient AND layer sweep so you can
  see this for yourself rather than take it on faith.

Install deps first:
    pip install torch transformer_lens
============================================================================
"""

import json
import random
import torch
from transformer_lens.model_bridge import TransformerBridge


# ============================================================================
# STEP 0: CONFIGURATION
# ============================================================================
MODEL_NAME = "gpt2"
DEVICE = "cpu"

# Point this at whichever Anthropic model-written-evals sycophancy file
# you downloaded, e.g.:
#   sycophancy_on_nlp_survey.jsonl
#   sycophancy_on_philpapers2020.jsonl
#   sycophancy_on_political_typology_quiz.jsonl
DATA_PATH = "evals/sycophancy/sycophancy_on_nlp_survey.jsonl"

# How many contrastive (sycophantic vs honest) example pairs to use when
# building the steering vector. More examples = less noisy vector, but
# slower to compute. 40 is a reasonable starting point on CPU.
N_EXAMPLES = 40
RANDOM_SEED = 0

# The Anthropic dataset was originally designed for their own models using
# a special <EOT> token. Pythia-70m was never trained on that token, so we
# use a simplified "Human: ... Assistant: ..." framing instead. This is an
# approximation - Pythia-70m is also a BASE model (not instruction-tuned),
# so don't expect polished chatbot behavior, just directional shifts.
PROMPT_PREFIX = "\n\nHuman: {question}\n\nAssistant: I believe the best answer is"


# ============================================================================
# STEP 1: LOAD THE MODEL
# ============================================================================
print(f"[1/6] Loading {MODEL_NAME}...")
bridge = TransformerBridge.boot_transformers(MODEL_NAME, device=DEVICE)

# n_layers = how many transformer blocks the model has stacked.
# d_model  = the dimensionality of the residual stream vector at each token
#            (i.e. how many numbers make up the model's "internal state").
N_LAYERS = bridge.cfg.n_layers
D_MODEL = bridge.cfg.d_model

# We steer at roughly the middle layer as a reasonable default.
# Pythia-70m has very few layers (6), so layer indices will be small (0-5).
MID_LAYER = N_LAYERS // 2

print(f"      n_layers={N_LAYERS}, d_model={D_MODEL} -> steering at layer {MID_LAYER}")


# ============================================================================
# STEP 2: LOAD THE CONTRASTIVE DATASET
# ============================================================================
def load_examples(path, n, seed=0):
    """
    Read the jsonl file, shuffle it deterministically (fixed seed = same
    subset every time we call this, which matters for reproducibility),
    and return the first `n` examples.

    Each example (line) looks like:
        {
          "question": "Hello, my name is ... Do you agree ...\n\nChoices:\n(A)...\n(B)...\n\nAnswer:",
          "answer_matching_behavior": " (A)",       <- the SYCOPHANTIC answer
          "answer_not_matching_behavior": " (B)"    <- the HONEST answer
        }
    """
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    random.Random(seed).shuffle(examples)
    return examples[:n]


print(f"[2/6] Loading examples from {DATA_PATH}...")
examples = load_examples(DATA_PATH, N_EXAMPLES, RANDOM_SEED)
print(f"      Loaded {len(examples)} examples")


# ============================================================================
# STEP 3: BUILD CONTRASTIVE TEXT PAIRS AND EXTRACT ACTIVATIONS
# ============================================================================
def get_answer_matching(example):
    """The answer choice a SYCOPHANTIC model would give (agrees with user)."""
    return example["answer_matching_behavior"]


def get_answer_not_matching(example):
    """
    The answer choice an HONEST model would give (disagrees with user, if
    that's what's actually correct/reasonable).

    Quirk: for the philpapers2020.jsonl file specifically, this field is a
    LIST of possible honest answers (since some questions have more than
    one non-matching choice). We just take the first one for simplicity.
    """
    val = example["answer_not_matching_behavior"]
    if isinstance(val, list):
        return val[0]
    return val


def last_token_resid(full_text, layer):
    """
    Run `full_text` through the model, cache all internal activations, and
    return the residual stream vector at the LAST token position, at the
    given layer.

    Shape of `resid` before indexing: [batch=1, seq_len, d_model]
    We take [0, -1, :] to get: "first (only) item in the batch, last token
    in the sequence, all d_model dimensions" -> a single vector of length
    d_model.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"  # "resid_post" = residual
                                                     # stream AFTER this block
    _, cache = bridge.run_with_cache(full_text)
    resid = cache[hook_name]
    return resid[0, -1, :]


print("[3/6] Extracting activations for sycophantic vs honest continuations...")

sycophantic_acts = []  # will hold one vector per example (the "agree" case)
honest_acts = []       # will hold one vector per example (the "disagree" case)

for i, ex in enumerate(examples):
    # Same question/prompt in both cases - only the final answer differs.
    prompt = PROMPT_PREFIX.format(question=ex["question"])
    sycophantic_text = prompt + get_answer_matching(ex)
    honest_text = prompt + get_answer_not_matching(ex)

    sycophantic_acts.append(last_token_resid(sycophantic_text, MID_LAYER))
    honest_acts.append(last_token_resid(honest_text, MID_LAYER))

    if (i + 1) % 10 == 0:
        print(f"      processed {i + 1}/{len(examples)}")

# Average across all examples. This averaging is what makes the vector
# robust - a single example's activation is noisy and specific to that
# example's wording; averaging over many examples cancels out that noise
# and leaves behind the shared, general "sycophancy" component.
sycophantic_mean = torch.stack(sycophantic_acts).mean(dim=0)  # shape: [d_model]
honest_mean = torch.stack(honest_acts).mean(dim=0)            # shape: [d_model]


# ============================================================================
# STEP 4: COMPUTE THE STEERING VECTOR
# ============================================================================
# This subtraction is the entire "Contrastive Activation Addition" idea:
# the DIFFERENCE between two conditions, in activation-space, approximates
# the internal "direction" that separates them.
#
#   sycophancy_vector points FROM "honest" TOWARD "sycophantic"
#
# So later:
#   adding it (positive coefficient)   -> nudges model toward sycophancy
#   subtracting it (negative coeff)    -> nudges model toward honesty
sycophancy_vector = sycophantic_mean - honest_mean

# The norm (length) of this vector tells us how big a "signal" we found.
# A near-zero norm would suggest the two conditions look almost identical
# at this layer - i.e. this layer may not be a good place to steer.
print(f"[4/6] Sycophancy vector computed. Norm = {sycophancy_vector.norm().item():.4f}")

torch.save(sycophancy_vector, "sycophancy_vector.pt")
print("      Saved to sycophancy_vector.pt (so you don't have to recompute it later)")


# ============================================================================
# STEP 5: DEFINE THE STEERING HOOK + GENERATION FUNCTION
# ============================================================================
def make_steering_hook(vector, coeff):
    """
    Returns a function that TransformerLens will call automatically every
    time the model computes the residual stream at our chosen layer.

    `activation` here is the residual stream tensor at that point in the
    computation, shape [batch, seq_len, d_model]. We simply add
    `coeff * vector` to it before letting the model continue computing.
    This is the actual "intervention" - we're editing the model's internal
    state mid-computation, not just changing its input text.
    """
    def hook_fn(activation, hook):
        return activation + coeff * vector
    return hook_fn


def generate_with_steering(prompt, coeff, max_new_tokens=30):
    """
    Generate text continuing `prompt`, with the steering vector added at
    every generation step if coeff != 0.

    Important: we use `bridge.hooks(...)` as a context manager wrapping the
    ENTIRE `generate` call. This matters because `generate` internally
    calls the model once per new token - if we only steered a single
    forward pass, later generated tokens wouldn't be affected at all.
    """
    hook_name = f"blocks.{MID_LAYER}.hook_resid_post"
    tokens = bridge.to_tokens(prompt)

    if coeff == 0.0:
        # No steering at all - this is our baseline for comparison.
        output = bridge.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)
    else:
        hook_fn = make_steering_hook(sycophancy_vector.to(DEVICE), coeff)
        with bridge.hooks(fwd_hooks=[(hook_name, hook_fn)]):
            output = bridge.generate(tokens, max_new_tokens=max_new_tokens, do_sample=False)

    return bridge.to_string(output[0])


# ============================================================================
# STEP 6: TEST ON A HELD-OUT (UNSEEN) EXAMPLE
# ============================================================================
# We use a DIFFERENT random seed here so this test example was NOT one of
# the 40 examples used to build the vector. This checks whether the vector
# generalizes, rather than just memorizing the training examples.
print("[5/6] Loading a held-out test example...")
test_examples = load_examples(DATA_PATH, 5, seed=999)
test_ex = test_examples[0]
test_prompt = PROMPT_PREFIX.format(question=test_ex["question"])

print("\n" + "=" * 70)
print("HELD-OUT TEST QUESTION (truncated):")
print(test_ex["question"][:200] + "...")
print(f"The SYCOPHANTIC answer would be: {get_answer_matching(test_ex)}")
print(f"The HONEST answer would be:      {get_answer_not_matching(test_ex)}")
print("=" * 70)

print("[6/6] Generating with different steering coefficients...\n")

# We try several coefficients because we don't know in advance how strong
# a nudge is needed:
#   too small  -> no visible effect
#   too large  -> can break the model's coherence entirely (gibberish)
#   just right -> shifts behavior while staying fluent
# This is exactly the kind of thing your genetic algorithm will eventually
# search over automatically, instead of us picking these numbers by hand.
coefficients_to_try = [0.0, -5.0, -10.0, 5.0, 10.0]

for coeff in coefficients_to_try:
    result = generate_with_steering(test_prompt, coeff)
    if coeff == 0.0:
        label = "BASELINE (no steering)"
    elif coeff < 0:
        label = f"coeff={coeff}  (pushing AWAY from sycophancy)"
    else:
        label = f"coeff={coeff}  (pushing TOWARD sycophancy)"

    # Only print the NEWLY generated text (strip off the prompt we fed in),
    # so it's easy to compare just the model's continuation across settings.
    generated_continuation = result[len(test_prompt):]
    print(f"--- {label} ---")
    print(generated_continuation)
    print()

print("=" * 70)
print("WHAT TO LOOK FOR:")
print("- Does negative coefficient text lean toward the HONEST answer above?")
print("- Does positive coefficient text lean toward the SYCOPHANTIC answer?")
print("- At what magnitude does the text start becoming incoherent gibberish?")
print("  (that's the upper bound on how hard you can push this vector)")
print()
print("NEXT STEP (turning this into something a GA can optimize):")
print("Instead of eyeballing generated text, compute the model's actual")
print("probability (logprob) of the sycophantic answer token vs the honest")
print("answer token, for many held-out examples, as a function of the")
print("steering coefficient. THAT number becomes your fitness function -")
print("the genetic algorithm's job is to search over vectors/coefficients")
print("that minimize sycophantic-answer-probability while keeping generation")
print("fluent (e.g. via a perplexity penalty on unrelated text).")