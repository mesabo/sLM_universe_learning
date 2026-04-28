# Exercises — Course 0 · ch3 · class 1

## 1. Warm-up — print the chat template

In a Python shell:
```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
print(t.chat_template)
```
Find where `system` / `user` / `assistant` roles are handled. What does the template do if `messages[0]` is *not* a system message?

## 2. Apply — count tokens for code vs prose

Create two prompts of the same character length: one English sentence, one Python snippet. Tokenize both with SmolLM2-135M's tokenizer and MiniLM's. Which tokenizer is more efficient on code? Why?

## 3. Stretch — render a tool-use chat

ChatML supports tool calls. Construct a `messages` list where the assistant calls a tool (`role: "tool"`). Render with `apply_chat_template`. Does SmolLM2-Instruct's template handle `role: "tool"`? If not, what's the smallest patch you'd need?
