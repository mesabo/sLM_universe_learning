# Exercises — Class 3: LangGraph Stateful Graphs

## Warm-up: Inspect the compiled graph

After `app = graph.compile()`, call `app.get_graph().draw_mermaid()` to generate a Mermaid diagram of the graph topology. Print it. Verify that all nodes and edges you defined appear in the diagram. Explain what the `__start__` and `__end__` nodes represent.

## Apply: Add a retry counter node

Add a `retry_count: int` field to `ReviewState`. Add a `retry` node that increments `retry_count` by 1 and a conditional edge after `classify_sentiment` that routes to `retry` if the model output is invalid, or to `assign_category` if valid. Set a maximum of 3 retries before forcing the graph to end. Verify that the counter is correctly incremented when you feed an ambiguous review.

## Stretch: Human-in-the-loop checkpoint

Add a `human_approved: bool` field to the state. After `classify_sentiment`, add a node `request_approval` that prints the classification and sets `human_approved = True` (simulating instant approval). Use `graph.compile(checkpointer=MemorySaver())` to enable checkpointing. Invoke the graph, pause at the approval node with `interrupt_before=["request_approval"]`, manually update the state via `app.update_state(config, {"human_approved": True})`, and resume. Verify the graph completes correctly after the manual update.
