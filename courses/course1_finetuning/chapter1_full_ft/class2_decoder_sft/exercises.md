# Exercises — Course 1 · ch1 · class 2

## 1. Warm-up — generate before & after

Before `bash run.sh`, copy the model card example completion. After the smoke run, `model.generate(...)` the same prompt from the saved checkpoint. What changes? What stays the same?

## 2. Apply — bigger backbone

Run with `backbone=HuggingFaceTB/SmolLM2-360M-Instruct`. How does eval loss change? How does GPU memory change?

## 3. Stretch — turn on packing

Set `train.packing: true` in the YAML. TRL packs multiple short examples into one sequence to use the full `max_seq_length`. Compare wall-clock training time. When does packing hurt (hint: when sequences vary wildly in length and attention mask logic matters)?
