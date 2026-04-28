# Notes — Course 0 · ch 3 · class 1 (tokenization & chat templates)

Use this file to capture observations from the lesson and to answer the exercises in `exercises.md`.

## Run log

| Run | Encoder tokens | Decoder tokens | chat_no_gen | chat_with_gen | Notes |
|---|---|---|---|---|---|
| default |  |  |  |  |  |

## Exercises

### 1. Warm-up — print the chat template

(What does SmolLM2-Instruct's Jinja template do for `system` / `user` / `assistant` roles? What if `messages[0]` is not a system message?)

### 2. Apply — code vs prose

(Token counts for the code snippet vs the English sentence with each tokenizer. Which is more efficient on code? Why?)

### 3. Stretch — render a tool-use chat

(Did SmolLM2-Instruct's template handle `role: "tool"`? If not, what's the smallest patch you'd need?)

## Open questions

- (Things that surprised you, or that you'd want to dig into further.)
