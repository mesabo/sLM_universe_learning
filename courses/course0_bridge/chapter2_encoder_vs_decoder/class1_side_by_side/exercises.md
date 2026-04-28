# Exercises — Course 0 · ch2 · class 1

## 1. Warm-up — change the prompt

Edit `configs/default.yaml`'s `prompt` to a question the encoder *cannot* answer (e.g. "Translate to French: hello"). Re-run. Note: the encoder still produces a vector — what does that vector "mean" here?

## 2. Apply — sample vs greedy

Set `generation.do_sample: true` and `generation.temperature: 0.8`. Re-run twice. Why are outputs different? What happens if you `set_seed(42)` *inside* `_decoder_step` before `model.generate(...)`?

## 3. Stretch — pool a decoder

Modify `_decoder_step` to ALSO compute a mean-pooled hidden-state vector (`output_hidden_states=True`). Compare its dimension and L2 norm to the MiniLM vector. What's the catch with using a 135M-param decoder as an encoder?
