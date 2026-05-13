# Notes — Course 0 · ch 2 · class 1 (encoder vs decoder side-by-side)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Encoder | Decoder | dim | tokens_generated | Notes |
|---|---|---|---|---|---|
| default | MiniLM | SmolLM2-135M-Instruct |  |  |  |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — change the prompt

(How did the encoder respond to a prompt it "couldn't answer"? What does its vector mean in that context?)

- The coder respond as `tokens_generated: 1` because it is a vector. It can only generate vectors and not sequential text.


### 2. Apply — sample vs greedy

(Why are sampled outputs different across runs? What happens when you `set_seed(42)` inside `_decoder_step`?)

- The output is different because of non deterministic computation.
- When seed is fixed, the clock becomes deterministic and output tend to be consistantly same.

### 3. Stretch — pool a decoder

(How did the mean-pooled SmolLM2 hidden-state vector compare to MiniLM's? What's the catch with using a 135M decoder as an encoder?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)

> Everything! The code wasn't working because of wrong implementations in `enc_metrics = _encoder_step(cfg["encoder_backbone"], prompts, n_passes, warmup, log)`

How did I solved it?

✦ 

  On line 85, the code originally looked like this:

   1 # The bug: It was just a tuple assigned to a variable, missing the function name
   2 `enc_metrics = (cfg["encoder_backbone"], prompts, n_passes, warmup, log)`

  I changed it to:

   1 # The fix: I added the function call to _encoder_step
   2 `enc_metrics = _encoder_step(cfg["encoder_backbone"], prompts, n_passes, warmup, log)`

  Without the `_encoder_step` function name, Python was simply creating a "Tuple" (a list of values) and saving it into enc_metrics. Later, when the code tried to use enc_metrics to log
  results or calculate stats, it would have crashed because you can't get "metrics" out of a simple list of strings.
