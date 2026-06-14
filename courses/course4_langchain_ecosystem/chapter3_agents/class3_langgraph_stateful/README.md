# Class 3 — LangGraph: Stateful Multi-Actor Graphs

## Psycho Mode

AgentExecutor is a linear loop — one model, one set of tools, step by step until done. LangGraph breaks this linearity. A LangGraph application is a directed graph where each node is a function that reads and updates a shared state, and edges define the flow of control. Conditional edges enable branching: the graph takes a different path depending on what the state contains. Cycles enable loops: the graph revisits a node until a stopping condition is met.

Think of it as a flowchart you can actually run. The "state" is a typed dictionary shared across all nodes. Each node is a pure function that transforms the state. The graph compiler produces an object you call like a regular function — but internally it routes through the graph topology, parallelizing independent paths where possible.

## Academic Mode

Formally, a LangGraph application is a directed graph $G = (V, E)$ where:
- $V = \{v_1, \ldots, v_n\}$ is a set of node functions, each $v_i : S \to S$ mapping state to state
- $E$ includes regular edges $(v_i, v_j)$ and conditional edges $(v_i, \phi_i)$ where $\phi_i : S \to V$ is a routing function
- A global state $s \in S$ (a TypedDict) is threaded through each node

Execution proceeds from the entry point, applying each node to the state in order, following edges until the `END` sentinel is reached. For checkpointing and human-in-the-loop, LangGraph supports state serialization at each step. Reference: LangGraph documentation — [https://langchain-ai.github.io/langgraph/](https://langchain-ai.github.io/langgraph/).

## Engineering Mode

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class ReviewState(TypedDict):
    text: str
    sentiment: str
    category: str
    done: bool

def classify_sentiment(state: ReviewState) -> ReviewState:
    # Call LLM, return updated state
    return {**state, "sentiment": "positive"}

def assign_category(state: ReviewState) -> ReviewState:
    mapping = {"positive": "praise", "negative": "complaint", "neutral": "inquiry"}
    return {**state, "category": mapping[state["sentiment"]], "done": True}

def route(state: ReviewState) -> Literal["assign_category", END]:
    return "assign_category" if state["sentiment"] else END

graph = StateGraph(ReviewState)
graph.add_node("classify_sentiment", classify_sentiment)
graph.add_node("assign_category", assign_category)
graph.set_entry_point("classify_sentiment")
graph.add_conditional_edges("classify_sentiment", route)
graph.add_edge("assign_category", END)

app = graph.compile()
result = app.invoke({"text": "Great product!", "sentiment": "", "category": "", "done": False})
```

Gotchas:
- All TypedDict fields must be present in the initial state dict passed to `invoke()`.
- Node functions must return a dict (full state or partial update); returning `None` causes a runtime error.
- Conditional edge functions must return a string matching an existing node name or `END`.
- For cycles, always include a counter field in state and check it in the routing function to prevent infinite loops.

Config keys: `backbone`, `max_graph_steps`, `limits.smoke.n_runs`.

## Research Mode

LangGraph is designed for multi-agent systems where different specialized agents run as nodes. Research directions: (1) dynamic graph construction — building the graph topology from a high-level task description rather than hardcoding it; (2) parallel node execution — when two nodes have no data dependency, running them concurrently on separate GPUs; (3) sub-graph composition — embedding one StateGraph inside another as a node, enabling hierarchical agent systems; (4) LangGraph Cloud — a managed runtime for stateful graph persistence with Redis or PostgreSQL backends.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter3_agents/class3_langgraph_stateful/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter3_agents/class3_langgraph_stateful/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `graph_compiled_ok` | [1, 1] |
| `routing_ok` | [1, 1] |
| `state_persisted_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
