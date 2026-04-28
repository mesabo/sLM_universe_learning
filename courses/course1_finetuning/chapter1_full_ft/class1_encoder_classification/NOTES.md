# Notes — Course 1 · ch 1 · class 1 (full FT encoder classification)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Backbone | mode | accuracy | f1_macro | Notes |
|---|---|---|---|---|---|
| default | sentence-transformers/all-MiniLM-L6-v2 | smoke |  |  |  |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — sweep the backbone

(MiniLM vs BGE-small vs GTE-small accuracy on AG News. Which wins, and is the gap meaningful at this scale?)

### 2. Apply — linear probe baseline

(How much accuracy did you lose with `freeze_backbone=true`? How much memory did you save?)

### 3. Stretch — different head architecture

(Did the deeper head help on AG News? Why or why not? Hint: encoder is doing most of the work.)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
