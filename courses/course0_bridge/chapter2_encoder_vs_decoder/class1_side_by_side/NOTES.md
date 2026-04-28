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

### 2. Apply — sample vs greedy

(Why are sampled outputs different across runs? What happens when you `set_seed(42)` inside `_decoder_step`?)

### 3. Stretch — pool a decoder

(How did the mean-pooled SmolLM2 hidden-state vector compare to MiniLM's? What's the catch with using a 135M decoder as an encoder?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
