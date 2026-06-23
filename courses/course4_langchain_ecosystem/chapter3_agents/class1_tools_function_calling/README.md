# Class 1 — Tools and Function Calling

> Goal: expose real Python functions to an LLM as typed tools, then show how schema-driven function calling turns a text generator into a system that can actually act on the outside world.

## Psycho Mode

An LLM on its own can only produce text. Tools break that constraint: they let the model invoke real Python functions, read databases, query APIs, or run shell commands. The model decides which tool to call and with what arguments; your code executes it and passes the result back. This turn-taking between model and tools is the foundation of all LLM agents.

Think of tools as a "skill registry." You define what each skill does (in a docstring), what inputs it takes (via Pydantic schema), and what it returns. The model reads the registry and selects the right tool for each subtask. The `@tool` decorator in LangChain automates the boilerplate: it wraps a Python function in a schema the model can read.

## Academic Mode

Let $\mathcal{T} = \{t_1, \ldots, t_k\}$ be a set of tools, where each $t_j$ has a name $n_j$, description $d_j$, and argument schema $\mathcal{A}_j$. The LLM, conditioned on user query $q$ and tool descriptions $\mathcal{T}$, outputs either a tool call $(t_j, a_j)$ where $a_j \in \mathcal{A}_j$, or a final text response. Tool selection is modeled as:

$$\hat{t}, \hat{a} = \arg\max_{t \in \mathcal{T}, a \in \mathcal{A}_t} p_\theta(\text{call}(t, a) \mid q, \mathcal{T})$$

The executor then computes $r = t(\hat{a})$ and feeds $r$ back to the model as an observation, starting the next reasoning step. OpenAI function calling formalizes this interface in a JSON schema appended to the API request. LangChain's `@tool` decorator builds compatible schemas automatically. Reference: [https://python.langchain.com/docs/concepts/tools/](https://python.langchain.com/docs/concepts/tools/).

## Engineering Mode

```python
from langchain_core.tools import tool

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b

# Schema inspection
print(add.name)         # "add"
print(add.description)  # "Add two numbers and return the result."
print(add.args_schema)  # Pydantic model with fields a: float, b: float

# Direct invocation (no LLM needed)
result = add.invoke({"a": 3.0, "b": 4.0})  # 7.0

# Bind tools to an LLM (enables tool calling in responses)
llm_with_tools = llm.bind_tools([add, multiply, sqrt])
response = llm_with_tools.invoke("What is 3 + 4?")
if response.tool_calls:
    for call in response.tool_calls:
        tool_result = add.invoke(call["args"])
```

Gotchas:
- Small local models (<500M) often do not generate tool call JSON even when tools are bound. Test direct invocation to confirm tool logic is correct before testing LLM-driven selection.
- The `@tool` decorator infers the schema from type hints and the docstring. Missing type hints or vague docstrings reduce tool selection accuracy.
- `bind_tools` is only meaningful for models that support function calling. `HuggingFacePipeline` does not support it natively; use `ChatHuggingFace` as the wrapper.

Config keys: `backbone`, `limits.smoke.n_prompts`.

## Research Mode

Tool use is a rapidly evolving area. Key advances: (1) Toolformer (Schick et al., 2023) — a self-supervised approach for teaching models when and how to call tools from raw text; (2) ToolBench (Qin et al., 2023) — a large-scale benchmark with 16,000+ real-world APIs; (3) multi-tool chaining — where the output of one tool is the input of another, creating implicit computation graphs. The current open challenge is tool selection accuracy at scale: when the registry has hundreds of tools, how does the model efficiently select the right one without reading all descriptions?

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter3_agents/class1_tools_function_calling/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter3_agents/class1_tools_function_calling/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `tool_direct_ok` | [1, 1] |
| `tool_schema_ok` | [1, 1] |
| `tool_bind_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
