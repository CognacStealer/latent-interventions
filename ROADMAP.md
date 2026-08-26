# Warm-up projects and reading list

A ladder up to *Evolutionary Discovery of Latent Interventions*. Each rung is
small enough to finish, and each one produces a piece of machinery the main
project needs. Nothing here is throwaway.

---

## Why warm up at all

The main project stacks four unfamiliar things on top of each other:

1. hooking into a transformer's internals
2. measuring behaviour with log-probabilities instead of reading text
3. sparse autoencoders as a feature basis
4. evolutionary search over that basis

If any number comes out wrong, you won't know which of the four caused it.
The warm-ups exist to make 1 and 2 boringly familiar, so that when you get to
3 and 4 there is only one new thing in the room.

---

## Rung 1 — Logit lens (one evening, ~40 lines)

**Question:** where inside GPT-2 does the answer actually get decided?

Take one prompt with an obvious completion — `"The Eiffel Tower is in the city of"`.
Run it with `run_with_cache`. At every layer, take the residual stream at the
last token, apply the model's final layer-norm, multiply by the unembedding
matrix `W_U`, and print the top-5 tokens.

You will watch `" Paris"` climb from nowhere to rank 1 somewhere in the middle
of the network.

**What it teaches:** what the residual stream *is* — not "hidden state" as an
abstract phrase, but a running draft of the output that every layer edits.
Once you have seen this, "add a vector to the residual stream at layer 6" stops
being magic and becomes obviously reasonable.

**What it gives the main project:** the intuition for choosing which layer to
steer at, and a diagnostic you'll reuse constantly.

---

## Rung 2 — The sentiment knob (a weekend, ~120 lines) ← recommended start

**Question:** can I build a steering vector where I already know the right answer?

The trouble with sycophancy is that you cannot tell by eye whether an output
got less sycophantic or just less coherent. Your own `results.txt` shows this:
at coeff −10 the model didn't become honest, it became *evasive*. Sentiment has
no such ambiguity.

1. Handwrite ~20 prompt pairs differing only in sentiment:
   `"I loved the film because it was"` / `"I hated the film because it was"`.
2. Mean-difference vector at each layer (same recipe as `steer.py`).
3. Score with log-probabilities, not generation: after a neutral prompt
   `"The film was"`, measure `logP(" great") − logP(" terrible")`.
4. Sweep **layer × coefficient** and plot the surface as a heatmap.

**Success criterion you can check without trusting anyone:** the heatmap should
show a band of layers where the effect is strong, weak effects at layer 0 and
the last layer, and a coefficient magnitude past which the model degenerates.

**What it teaches:** that steering is layer-specific, that there is a coherence
cliff, and — most importantly — the habit of validating a method on a problem
where you know the answer before pointing it at one where you don't.

**What it gives the main project:** the layer×coefficient sweep is a direct
rehearsal of what you'll do with `evaluate.py`, on data you can sanity-check.

---

## Rung 3 — Activation patching on a toy circuit (a week)

**Question:** which *component* causes a behaviour, not just which direction?

Reproduce a slice of the IOI (indirect object identification) result. Prompt:
`"When John and Mary went to the shop, John gave a drink to"` → `" Mary"`.
Corrupted prompt: swap the names. Then, one component at a time, copy the
clean activation into the corrupted run and measure how much of the correct
answer's logit is restored.

**What it teaches:** causal intervention as the core method of the field.
Correlation (a probe can read X off layer 8) versus causation (editing layer 8
changes the behaviour) is the distinction the whole discipline turns on.

**What it gives the main project:** patching is how you'll later *verify* that
an evolved genome works for the reason you think it does, rather than by some
side effect. Without this step, a winning genome is an unexplained artifact.

---

## Rung 4 — One SAE, by hand (a few days)

**Question:** what does the feature basis actually look like?

Load a single GPT-2-small residual SAE via SAELens. Run your sycophancy prompts
through it. For the top ~10 firing features at the decision token, look each one
up on Neuronpedia and read its explanation. Then: take the difference in feature
activations between the sycophantic and honest versions, and see which features
separate them.

**What it teaches:** whether the sparse basis is telling you something the raw
768-dim vector wasn't. Sometimes it is. Sometimes the top features are
`" ("` and `"A"` and you learn your prompt format is doing the work — which is
itself a finding worth catching before it contaminates a paper.

**What it gives the main project:** this *is* the genome's search space. You
cannot design mutation and crossover over a space you have never looked at.

---

## Then: the main project

Rungs 1–4 give you hooks, log-prob metrics, causal verification, and the
feature basis. What's left is genuinely the new contribution: a search
procedure over `G = {F, L, A, C}` with `evaluate.py` as fitness.

---

# Reading order

You cannot front-load this field. Reading twenty papers and then starting to
code produces the feeling of understanding without the substance — the ideas
are simple to state and hard to believe until you have watched them happen in
a tensor. So the order below **interleaves reading with the rungs above**.
Seven papers, four sessions of reading, two coding stops.

| # | Paper | Read | Then |
|---|-------|------|------|
| 1 | A Mathematical Framework for Transformer Circuits | §"Residual stream as a communication channel" only | **Rung 1: logit lens** |
| 2 | Activation Addition (ActAdd) | Whole thing, it's short | — |
| 3 | Steering Llama 2 via Contrastive Activation Addition | Whole thing, especially the evaluation section | **Rung 2: sentiment knob, then run `evaluate.py`** |
| 4 | Analysing the Generalisation and Reliability of Steering Vectors | Whole thing | — |
| 5 | Towards Monosemanticity | "Problem Setup" + skim the feature walkthroughs | — |
| 6 | Improving Steering Vectors by Targeting SAE Features | Whole thing | **Rung 4: one SAE by hand** |
| 7 | Sparse Feature Circuits **and** ACDC | Method sections; compare their search to yours | Design the genome |

Stop at #3 and write code. Papers 1–3 are enough to build everything up to and
including the sweep, and the sweep will teach you things papers 4–7 assume you
already noticed.

## What each one is for

1. **Framework** — reframes the transformer as a residual stream that every
   layer reads from and writes to. Without this, "add a vector at layer 6" is
   arbitrary; with it, it is obvious.
2. **ActAdd** — the core move: no gradients, just a difference of activations.
3. **CAA** — *this is what `steer.py` already implements*, sycophancy dataset
   included. Read their evaluation methodology closely; `evaluate.py` is
   reconstructing it.
4. **Tan et al.** — the corrective. Steering vectors are far more brittle than
   the headline results suggest. Read it once you have your own numbers, so it
   lands as a warning rather than as discouragement.
5. **Monosemanticity** — why raw neurons are uninterpretable (superposition)
   and what an SAE fixes. This is the paper that makes "SAE feature basis"
   mean something.
6. **Chalnev, Siu & Conmy** — steering vectors constructed to target *specific
   SAE features*. This is the direct precursor to your SAE-basis genome:
   they hand-pick the target features, you would evolve them.
7. **Sparse Feature Circuits + ACDC** — the two nearest neighbours to your
   thesis. Read them as a reviewer would: what does evolutionary search do
   that greedy pruning and gradient attribution do not?

---

# Reading list

Full details. Papers marked ✅ are already in your `RAG_Research` folder.

## Tier 0 — Foundations

- **A Mathematical Framework for Transformer Circuits** — Elhage et al., 2021.
  <https://transformer-circuits.pub/2021/framework/index.html>
  The residual stream as a communication channel that every layer reads from
  and writes to. This is the mental model everything else assumes. Read at
  least §"Residual stream as a communication channel".

- **In-context Learning and Induction Heads** — Olsson et al., 2022.
  <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>
  The best worked example of "we found a mechanism and proved it does
  something." Sets the standard for what a claim in this field looks like.

## Tier 1 — Steering and linear representations (core to your project)

- ✅ **Plug and Play Language Models** — Dathathri et al., 2019. arXiv:1912.02164
  `Plug and Play Models.pdf`. The ancestor: control by editing activations at
  inference. Gradient-based and clunky, but it framed the problem. Background,
  not required.

- ✅ **Extracting Latent Steering Vectors from Pretrained Language Models** —
  Subramani, Suresh & Peters, ACL Findings 2022.
  `Extract latent vectors .pdf` (**note: `Extracting steering vectors.pdf` is a
  byte-identical duplicate — delete one**). Finds steering vectors by
  optimisation rather than by contrast. Historical interest; ActAdd and CAA
  superseded the method. Skim.

- ✅ **Improving Steering Vectors by Targeting Sparse Autoencoder Features** —
  Chalnev, Siu & Conmy, 2024. `Improving steering vectors .pdf`
  (**note: `last_prompt.txt` is a byte-identical copy of this PDF under a wrong
  extension — delete it**).
  **The most important paper you already own, and you may not know it.** It
  constructs steering vectors designed to activate *chosen SAE features* while
  leaving others untouched — the exact bridge between the raw-direction basis
  you have now and the feature basis your genome needs. They select the target
  features by hand. Evolving that selection is your project in one sentence.
  Note the co-author: Arthur Conmy, who also wrote ACDC.

- **Activation Addition: Steering Language Models Without Optimization** —
  Turner et al., 2023. <https://arxiv.org/abs/2308.10248>
  ActAdd. Showed you can skip the gradients entirely and just add a
  difference-of-activations vector.

- **Steering Llama 2 via Contrastive Activation Addition** — Panickssery
  (Rimsky) et al., 2023. <https://arxiv.org/abs/2312.06681>
  **This is exactly what `steer.py` implements**, including the sycophancy
  dataset. Read it closely — their evaluation methodology is the thing
  `evaluate.py` is reconstructing.

- ✅ **Representation Engineering: A Top-Down Approach to AI Transparency** —
  Zou et al., 2023. <https://arxiv.org/abs/2310.01405>
  `Representation Engineering_...pdf`. The broad framing your proposal sits
  inside. Low information density per page — skim for vocabulary, don't study.

- **The Geometry of Truth** — Marks & Tegmark, 2023.
  <https://arxiv.org/abs/2310.06824>
  Careful treatment of when a concept really is a linear direction, and how to
  tell. Directly relevant to your proposal's "assumption of linear concept
  representations" limitation.

- **Refusal in Language Models Is Mediated by a Single Direction** — Arditi et
  al., 2024. <https://arxiv.org/abs/2406.11717>
  The single best template for the *whole pipeline* done rigorously: find a
  direction, ablate it, verify causally, check it generalises. If you copy one
  paper's methodology, copy this one.

- **Analysing the Generalisation and Reliability of Steering Vectors** — Tan et
  al., NeurIPS 2024. <https://arxiv.org/abs/2407.12404>
  The skeptical paper. Steering vectors are far less reliable than the
  headline results suggest — high variance across prompts, brittle out of
  distribution. **Read this before you believe your own results.** It is also
  the strongest argument *for* your project: if hand-picked vectors are
  unreliable, automated search over them is worth having.

## Tier 2 — Circuits and causal methods

- **Interpretability in the Wild: a Circuit for Indirect Object
  Identification in GPT-2 small** — Wang et al., 2022.
  <https://arxiv.org/abs/2211.00593>
  The reference circuit. Source material for Rung 3.

- **Locating and Editing Factual Associations in GPT** — Meng et al., 2022.
  <https://arxiv.org/abs/2202.05262>
  ROME. Introduced causal tracing — the patching methodology in its clearest
  form.

- **Towards Automated Circuit Discovery for Mechanistic Interpretability** —
  Conmy et al., 2023. <https://arxiv.org/abs/2304.14997>
  ACDC. **The closest existing work to your thesis**: automating the search
  that humans previously did by hand. Understand precisely how your
  evolutionary approach differs from their greedy pruning — that difference is
  your contribution, and a reviewer will ask.

## Tier 3 — Sparse autoencoders

- **Sparse Autoencoders Find Highly Interpretable Features in Language
  Models** — Cunningham et al., 2023. <https://arxiv.org/abs/2309.08600>

- **Towards Monosemanticity** — Bricken et al., Anthropic, 2023.
  <https://transformer-circuits.pub/2023/monosemantic-features/index.html>
  The clearest explanation of *why* superposition makes raw neurons
  uninterpretable and what an SAE fixes.

- **Scaling Monosemanticity** — Templeton et al., Anthropic, 2024.
  <https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html>
  Golden Gate Claude. Feature steering at production scale.

- **Sparse Feature Circuits** — Marks et al., 2024.
  <https://arxiv.org/abs/2403.19647>
  **The nearest neighbour to your §4.** They discover circuits made of SAE
  features and edit them. Read carefully to work out what evolutionary search
  adds that their gradient-based attribution does not.

- **Gemma Scope** — Lieberum et al., 2024. <https://arxiv.org/abs/2408.05147>
  Reference for what an open SAE suite provides, if you outgrow GPT-2.

## Tier 4 — The evolutionary half

- **Evolutionary Optimization of Model Merging Recipes** — Akiba et al., 2024.
  <https://arxiv.org/abs/2403.13187>
  Evolution over model-space rather than latent-space, but the closest
  existence proof that evolutionary search is a serious tool in this domain
  and not an eccentricity. Useful for framing your motivation section.

- ✅ Your `Optimization algos/` folder. A blunt opinion, offered because it
  will save you weeks: read **Particle Swarm Optimization** (Kennedy &
  Eberhart) and **Clerc & Kennedy**'s convergence analysis, which you have,
  and stop there. Firefly, Cuckoo Search, Grey Wolf and Bee Colony are widely
  criticised in the optimisation literature as the same handful of mechanisms
  redescribed with new animal metaphors — see Sörensen, *Metaheuristics: the
  metaphor exposed* (2015). A reviewer of a mech-interp paper will not be
  impressed that you tried six of them; they will ask why a plain GA or CMA-ES
  was not sufficient. Pick one well-understood optimiser and justify it.

## Not on this path

`RAG ANYTHING.pdf`, `SELF-RAG`, `ReAct`, `Topological Structure Learning`,
`download_doc_admin_student.pdf` and `Attention is all you need.pdf` are from
a different project (the folder is still called `RAG_Research`). Nothing wrong
with them — they are just not on the critical path for this one. Attention Is
All You Need in particular: if you can already say what a key, query and value
are, re-reading it will not help; paper #1 in the order above is the reframe
you actually need.

## Tools and courses

- **ARENA** — <https://arena.education/> — the standard hands-on mech-interp
  curriculum. The transformer-interpretability chapter covers Rungs 1–3 with
  worked exercises and solutions.
- **TransformerLens docs** — <https://transformerlensorg.github.io/TransformerLens/>
- **SAELens** — <https://github.com/jbloomAus/SAELens> — loading pretrained SAEs.
- **Neuronpedia** — <https://www.neuronpedia.org/> — feature explanations and
  dashboards. Use it to *interpret* results; run the search locally.
- **200 Concrete Open Problems in Mechanistic Interpretability** — Neel Nanda.
  Useful for calibrating what counts as a tractable project.

---

## A note on scope

Your proposal is ambitious — cross-model generalisation, reasoning circuits,
Pareto frontiers. That is fine for a proposal. But the version that gets
finished is much narrower:

> *One model (GPT-2 small). One behaviour (sycophancy). One genome
> representation (sparse SAE feature subsets at a single layer). One
> baseline to beat (the CAA vector). Evolve, then check whether the same
> features keep winning across independent runs.*

If that works, everything in the proposal is a follow-up. If it doesn't, you
found out cheaply.
