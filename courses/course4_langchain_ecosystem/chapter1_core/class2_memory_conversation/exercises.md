# Exercises — Class 2: Conversation Memory

## Warm-up: Count history messages

After simulating 4 conversation turns, print `len(history.messages)` and verify it equals 8 (4 human + 4 AI). Explain why the count is 2× the number of turns.

## Apply: Sliding window memory

Implement a sliding window by trimming `history.messages` to the last `2 * k` messages before each turn (keeping only the last `k` pairs). Set `k = 2` and run 6 turns. Verify that after the 6th turn, the history contains exactly 4 messages (the last 2 pairs), and that the model still correctly references the 5th turn's content if asked directly.

## Stretch: Multi-session isolation

Spawn two concurrent sessions (`"alice"` and `"bob"`) using separate session IDs. Have Alice say "My favorite color is blue" and Bob say "My favorite color is red." Then ask both "What is my favorite color?" Verify that Alice's chain returns "blue" and Bob's returns "red," demonstrating session isolation.
