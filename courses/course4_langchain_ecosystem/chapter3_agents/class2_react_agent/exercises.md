# Exercises — Class 2: ReAct Agent

## Warm-up: Log intermediate steps

Enable `verbose=True` on `AgentExecutor` and run one task. Read the printed trace. Identify: (1) how many Thought-Action-Observation cycles occurred, (2) which tool was selected and with what arguments, (3) how the model composed the final answer from the observation.

## Apply: Multi-step chain task

Create a task that requires two sequential tool calls: "What is the square root of (3 multiplied by 48)?" The agent must first call `multiply(3, 48)` to get `144`, then call `sqrt(144)` to get `12`. Verify `intermediate_steps` contains exactly two entries. If the agent skips a step, how would you diagnose and fix the issue?

## Stretch: Token budget guardrail

Add a token counter to the agent loop: before each step, count the total tokens in `intermediate_steps` using `tiktoken`. If the budget exceeds a threshold, inject a tool result `"BUDGET_EXCEEDED"` to force the agent to summarize and stop. Implement this as a custom `AgentExecutor` subclass that overrides `_return`.
