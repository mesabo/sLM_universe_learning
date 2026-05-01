# Notes — Course 3 · ch 4 · class 1 (PSI + canary)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | shift_schedule | psi_max | n_above_alarm | corr(psi, canary) | Notes |
|---|---|---|---|---|---|
| default | [0,0,.2,.4,.6,.8] |  |  |  |  |
|  |  |  |  |  |  |

## Exercises

### 1. Warm-up — sweep OOD ramp

(Did psi_max track the peak OOD fraction? When did the schedule become too flat for the alarm?)

### 2. Apply — refresh the canary

(Did accuracy_psi_correlation get tighter when the canary stayed representative?)

### 3. Stretch — MMD vs PSI

(Did MMD pick up drift earlier than PSI? At what cost?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
