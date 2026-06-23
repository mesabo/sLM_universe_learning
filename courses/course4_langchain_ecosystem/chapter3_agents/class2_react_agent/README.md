# Class 2 — ReAct Agent with AgentExecutor

> Goal: build a ReAct loop that reasons, calls tools, observes results, and continues until done, so the student sees how an agent differs from a single one-shot LLM call.

## Psycho Mode

ReAct stands for Reasoning and Acting interleaved. Instead of generating a single response, a ReAct agent loops: it reasons about what to do next, acts by calling a tool, observes the result, then reasons again. This loop continues until the agent decides it has enough information to produce a final answer. The key insight is that explicit reasoning steps (written out as "Thought:") dramatically improve tool selection accuracy compared to directly outputting tool calls.

Think of it like a chef following a recipe step by step, tasting as they go. They do not cook the whole dish blindly — they taste after each step and adjust. The ReAct loop is the agent's "taste and adjust" cycle.

## Academic Mode

The ReAct paradigm (Yao et al., 2023) generates a trajectory of interleaved thought, action, and observation steps:

$$\tau = (t_1, a_1, o_1, t_2, a_2, o_2, \ldots, t_T, a_T^{\text{final}})$$

where $t_i$ is a free-text thought, $a_i$ is a tool call (or final answer), and $o_i$ is the tool's return value. The LLM produces the next thought-action pair conditioned on the full trajectory so far:

$$t_{i+1}, a_{i+1} = f_\theta(\tau_{1:i})$$

AgentExecutor enforces an iteration limit $T_{\max}$ and parses each step into structured (tool, args) tuples using a `ReActSingleInputOutputParser`. The paper is: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models," ICLR 2023 — [https://arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629).

## Engineering Mode

```python
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

# Fetch the canonical ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)
result = executor.invoke({"input": "What is 15 + 27?"})
print(result["output"])              # final answer
print(result["intermediate_steps"])  # [(AgentAction, observation), ...]
```

For chat models (preferred for local sLMs), use `create_tool_calling_agent` with a `ChatPromptTemplate` instead:

```python
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)
```

Config keys: `backbone`, `limits.smoke.max_iterations`, `limits.smoke.n_tasks`.

Gotchas:
- Local sLMs struggle with ReAct format parsing. `handle_parsing_errors=True` is essential.
- `max_iterations` prevents infinite loops. Always set it.
- `hub.pull()` requires an internet connection; the class uses `create_tool_calling_agent` as the primary path.

## Research Mode

ReAct agents have a fundamental limitation: their reasoning is entirely serial. Tree of Thoughts (Yao et al., 2023) explores branching reasoning paths and selects the best. MCTS-based planners (AlphaCodium, etc.) use rollouts to evaluate intermediate steps. For production: (1) consider `max_execution_time` instead of `max_iterations` for latency-sensitive apps; (2) structured output at each step reduces parsing failures; (3) self-correction agents (Reflexion, Shinn et al., 2023) add a verification step after each trajectory to catch errors.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter3_agents/class2_react_agent/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter3_agents/class2_react_agent/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `agent_ok` | [1, 1] |
| `avg_steps` | [0.0, 10.0] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
