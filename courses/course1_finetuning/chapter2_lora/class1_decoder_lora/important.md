# Important Notes: LoRA and Projections

These notes are derived from the two main questions:

1. What is LoRA?
2. What are projections in transformer models?

---

## 1. LoRA Demystified

LoRA means **Low-Rank Adaptation**.

The core idea is simple:

> Instead of training the whole model, freeze the original model and train only tiny extra matrices attached to it.

So the base model stays unchanged, and LoRA learns a small correction on top.

## The Problem LoRA Solves

In normal full fine-tuning, we update all model weights.

For example, with a 135M parameter model, training may touch roughly:

```text
135,000,000 parameters
```

That is expensive because it needs more GPU memory, more storage, and more training time.

LoRA asks:

> Maybe we do not need to change the whole model. Maybe the useful task-specific change can be represented by a much smaller update.

So instead of changing the original weight matrix `W`, LoRA adds a small trainable update:

```text
W' = W + BA
```

Where:

```text
W  = original frozen weight matrix
A  = small trainable matrix
B  = small trainable matrix
BA = learned correction
```

The model behaves as if its weights changed, but the original weights never actually move.

## Simple Mental Model

Imagine the base model is a car engine.

Full fine-tuning rebuilds the engine.

LoRA keeps the engine exactly the same and adds a small steering/control module that learns how to guide it for your task.

So:

```text
Full fine-tuning = change the original model
LoRA             = keep the original model and train a small adapter
```

## Where LoRA Is Added

In transformer decoder models, LoRA is usually added to **linear layers inside attention**, such as:

```text
q_proj
k_proj
v_proj
o_proj
```

These are part of the attention mechanism:

```text
q_proj = query projection
k_proj = key projection
v_proj = value projection
o_proj = output projection
```

During training, the input goes through both:

```text
original frozen layer
+
LoRA adapter layer
```

Then their outputs are added together.

Instead of:

```text
output = W @ x
```

LoRA gives:

```text
output = (W + BA) @ x
```

Only `A` and `B` are trained.

## Why LoRA Is Efficient

Suppose a normal model matrix is:

```text
576 x 576 = 331,776 parameters
```

LoRA with rank `r = 16` uses two smaller matrices:

```text
A = 16 x 576
B = 576 x 16
```

Total:

```text
9,216 + 9,216 = 18,432 parameters
```

That is much smaller than `331,776`.

For the whole model, instead of training all 135M parameters, this class expects you to train only a small fraction, maybe around:

```text
~400K parameters
```

This is why the README says LoRA trains only around **0.3%** of the model.

## Important LoRA Terms

### Rank

`rank`, usually written as `r`, controls how large the adapter is.

Small rank:

```text
r = 4 or 8
```

Uses less memory, but has less learning capacity.

Larger rank:

```text
r = 32 or 64
```

Can learn more complex changes, but uses more memory.

### Alpha

`alpha` controls how strongly the LoRA update affects the base model.

A common rule is:

```text
alpha = 2 * rank
```

So if:

```text
rank = 16
```

Then:

```text
alpha = 32
```

## What The Training Code Usually Does

The training script usually does this:

1. Load the base model.
2. Freeze the base model parameters.
3. Create a `LoraConfig`.
4. Use `get_peft_model(...)` to inject LoRA adapters.
5. Train with `SFTTrainer`.
6. Save only the adapter weights.

The important part:

> The final saved file is not a full model. It is just the LoRA adapter.

At inference time, you load:

```text
base model + LoRA adapter
```

For example:

```text
SmolLM2-135M-Instruct + your trained LoRA adapter
```

## Why LoRA Is Useful

LoRA lets you have one base model and many small adapters:

```text
base model
  + math adapter
  + medical adapter
  + customer support adapter
  + Japanese assistant adapter
  + coding adapter
```

Each adapter may be only a few MB, instead of storing many full copies of the model.

## Common Confusion

Question:

> If the base model is frozen, how does it learn?

Answer:

> Because the model's forward pass changes.

The frozen base still computes its normal output, but LoRA adds a trainable correction.

So the model is not learning by changing `W`.

It learns by changing `A` and `B`, which create the update `BA`.

## One-Sentence Summary

LoRA fine-tuning means:

> Keep the original model frozen, attach small trainable adapter matrices to important layers, train only those adapters, and use them as a lightweight task-specific correction.

---

## 2. Projections Demystified

In this context, **projection** just means:

> A learned linear transformation.

A projection layer is usually just:

```text
y = W x + b
```

Or often without bias:

```text
y = W x
```

Where:

```text
x = input vector
W = learned weight matrix
y = transformed output vector
```

That is the whole basic idea.

## What Projection Means In Transformers

In a transformer, every token is represented as a vector.

For example, maybe each token has a hidden vector of size `576`:

```text
"Paris" -> [0.12, -0.44, 0.08, ...]  # length 576
```

The model uses projection layers to transform that same token vector into different roles.

For attention, those roles are:

```text
q_proj -> query
k_proj -> key
v_proj -> value
o_proj -> output
```

Think of the same hidden vector being passed through different learned lenses:

```text
hidden_state -> q_proj -> query vector
hidden_state -> k_proj -> key vector
hidden_state -> v_proj -> value vector
```

The word **projection** means:

> Take the original representation and map it into another representation space that is useful for a specific job.

## Query, Key, And Value

Attention asks:

```text
For this token, which other tokens should I pay attention to?
```

It does that using queries and keys.

```text
query = what I am looking for
key   = what I contain / how I can be matched
value = the information I will contribute if selected
```

Example sentence:

```text
The cat sat on the mat
```

When processing `"sat"`, the model may ask:

```text
Who did the sitting?
```

So `"sat"` gets a **query** vector.

The token `"cat"` has a **key** vector that may match that query strongly.

If the match is strong, the model pulls information from `"cat"`'s **value** vector.

So:

```text
q_proj makes search vectors
k_proj makes matchable label vectors
v_proj makes content vectors
```

After attention combines the values, `o_proj` maps the result back into the normal hidden-state space.

## Why LoRA Targets Projections

LoRA often attaches to:

```text
q_proj
k_proj
v_proj
o_proj
```

Because these layers strongly influence how the model routes information.

Instead of changing the original projection matrix:

```text
W
```

LoRA adds a small correction:

```text
W + BA
```

So for `q_proj`, for example:

```text
query = q_proj(hidden)
```

Conceptually becomes:

```text
query = original_q_proj(hidden) + lora_correction(hidden)
```

LoRA is therefore not teaching the whole model from scratch.

It is slightly changing how the model forms queries, keys, values, and outputs.

## Projection Summary

A projection is just a matrix that changes a vector from one useful form into another.

In transformers:

```text
q_proj: turns hidden state into "what am I looking for?"
k_proj: turns hidden state into "how can others find me?"
v_proj: turns hidden state into "what information do I provide?"
o_proj: turns attended information back into normal model language
```

## Final Combined Summary

LoRA works well on projection layers because projection layers control how information is searched, matched, carried, and returned inside attention.

Instead of retraining the whole model, LoRA learns small corrections to these projection layers.

That gives the model task-specific behavior while keeping the original model frozen.
