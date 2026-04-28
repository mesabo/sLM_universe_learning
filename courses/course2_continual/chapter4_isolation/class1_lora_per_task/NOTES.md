# Notes — Course 2 · ch 4 · class 1 (LoRA-per-task isolation)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | r | modules_to_save | acc_A_after_B | acc_B_after_B | BWT | Notes |
|---|---|---|---|---|---|---|
| default | 16 | [classifier] |  |  |  |  |
|  |  |  |  |  |  |  |

## Exercises

### 1. Warm-up — verify the head IS per-adapter

(Were the two adapters' classifier weights actually different? If not, `modules_to_save` isn't taking effect.)

### 2. Apply — drop `modules_to_save`

(BWT with `modules_to_save: []`. Did forgetting return? By how much?)

### 3. Stretch — three tasks

(Outline of the N-task generalization. Did BWT stay near 0?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
