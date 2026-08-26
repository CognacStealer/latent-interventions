# Evolutionary Discovery of Latent Interventions

Research notes and working code for the question: **can you *search* for steering
vectors instead of hand-building them?**

> The folder is called `RAG_Research` for historical reasons — it started as RAG
> reading and turned into interpretability work. The RAG papers are still in the
> reading list; the code is all steering and interpretability.

## The idea

Representation engineering shows you can change a model's behaviour at inference
time by adding a vector to its hidden activations:

```
h' = h + αv
```

where `h` is the residual stream, `v` a steering direction and `α` its strength.
Methods like ActAdd, Contrastive Activation Addition and SAE feature steering all
work — but they all get `v` the same way: **a human picks a contrastive concept
pair** and takes a difference of means.

That's the limitation I'm poking at. Hand-picked pairs mean:

- you only find directions you already thought to look for
- they transfer badly across models
- multi-feature interactions are essentially undiscoverable by hand
- everything assumes concepts are linearly represented

**The proposal:** treat the space of latent interventions as a search space and
let an evolutionary algorithm explore it. A genome encodes an intervention
(which layer, which features, what coefficients); fitness is measured
behaviourally; selection does the rest. That's what `Report.md` argues for.

## Why this needs a fitness function, and why that was the hard part

`steer.py` demonstrated a sycophancy direction exists — but it judged success by
*reading the generated text*. You cannot evolve a population of 200 genomes by
reading 200 paragraphs. Evolution needs a scalar.

`evaluate.py` is that scalar. Instead of generating anything, it asks the model
under intervention how much probability mass it puts on each of two candidate
answers:

```
margin = logP(sycophantic answer) − logP(honest answer)
```

Two numbers come out of it:

1. **Effect** — did the intervention move the behaviour we care about?
2. **Cost** — did it wreck the model's general fluency in the process?

Optimising effect alone finds interventions that make the model honest by making
it incoherent. The cost term is what stops the search cheating.

## Files

| File | What it is |
|---|---|
| `Report.md` | The review + research proposal. Start here |
| `ROADMAP.md` | A ladder of warm-up projects, each one building machinery the main project needs |
| `PAPERS.md` | Reading list with links (the PDFs themselves aren't committed) |
| `Latent_space.py` | Pulling hidden states out of TinyLlama-1.1B — the first "look inside a model" script |
| `Coding/pythia.py` | `transformer_lens` basics: `run_with_cache`, reading `blocks.4.hook_resid_post`, and a hook that adds a *random* direction to prove the plumbing works |
| `Coding/steer.py` | Contrastive Activation Addition on Pythia-70M — builds a sycophancy steering vector from paired agree/disagree completions and subtracts it at generation time |
| `Coding/evaluate.py` | The quantitative fitness function described above |
| `Coding/evals/sycophancy/` | The eval set |

## The roadmap, briefly

The main project stacks four unfamiliar things at once — hooking into a
transformer's internals, measuring behaviour via log-probabilities, sparse
autoencoders as a feature basis, and evolutionary search over that basis. If a
number comes out wrong you won't know which of the four caused it. So the
warm-ups exist to make (1) and (2) boring before (3) and (4) get added.

Rung 1 is the **logit lens**: take `"The Eiffel Tower is in the city of"`, run it
with `run_with_cache`, and at every layer apply the final layer-norm and the
unembedding to the last-token residual stream. You watch `" Paris"` climb from
nowhere to rank 1 somewhere in the middle of the network. After that, "add a
vector to the residual stream at layer 6" stops being magic.

## Running it

```bash
cd Coding
uv sync                    # transformer_lens, torch, transformers
python pythia.py           # sanity-check the hooks
python steer.py            # build and apply the sycophancy vector
python evaluate.py         # score an intervention numerically
```

Pythia-70M is small enough to run on CPU, slowly. The whole point of starting
there is that iteration speed beats scale while the machinery is still being
debugged.

## Status

Warm-ups and the fitness function are working. The evolutionary search loop
itself is still to be written — that's the next piece.
