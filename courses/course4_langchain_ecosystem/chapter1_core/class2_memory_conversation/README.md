# Class 2 — Conversation Memory

## Psycho Mode

Imagine talking to someone who forgets everything you said the moment you say it. Every message would have to re-introduce all context. That is exactly what a stateless LLM does by default: each call is independent. Conversation memory solves this by maintaining a "working notepad" of prior turns and prepending it to each new prompt.

The conceptual model: memory is a list of (role, content) pairs. At each turn, you append the user's message and the model's reply. When building the next prompt, you include this history. The model then sees the full context and can answer questions like "what did I say earlier?" because the answer is literally in its input. LangChain's memory layer automates this append-and-inject cycle.

## Academic Mode

Let $H_t = [(r_1, c_1), \ldots, (r_t, c_t)]$ be the conversation history at turn $t$, where $r_i \in \{\text{human}, \text{ai}\}$ and $c_i$ is the message content. The prompt at turn $t+1$ is constructed as:

$$p_{t+1} = \text{PromptTemplate}(H_t, q_{t+1})$$

where $q_{t+1}$ is the new user query. The LLM generates $a_{t+1} = f_\theta(p_{t+1})$, and the history is updated: $H_{t+1} = H_t \cup [(\text{human}, q_{t+1}), (\text{ai}, a_{t+1})]$.

Buffer memory stores all of $H_t$ verbatim (token cost grows linearly). Window memory stores only the last $k$ turns (bounded cost, may lose early context). Summary memory compresses old turns into a summary using a second LLM call (balanced cost and coverage). The LCEL-native API is `RunnableWithMessageHistory` documented at [https://python.langchain.com/docs/concepts/memory/](https://python.langchain.com/docs/concepts/memory/).

## Engineering Mode

The modern LCEL pattern for memory uses `InMemoryChatMessageHistory` and `RunnableWithMessageHistory`:

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])
chain = prompt | llm | StrOutputParser()

store = {}
def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
chain_with_history.invoke({"input": "Hello"}, config={"configurable": {"session_id": "s1"}})
```

Gotchas:
- Session IDs are arbitrary strings; using the same ID across two processes will not share state (history lives in-process by default).
- `MessagesPlaceholder` must match `history_messages_key` exactly.
- For multi-user apps, use a persistent backend (Redis, SQL) instead of `InMemoryChatMessageHistory`.

## Research Mode

Memory design is an open research problem. Transformers have a fixed context window; naive buffer memory will eventually exceed it. Recent approaches include: (1) hierarchical memory (short-term + long-term with retrieval), (2) MemGPT-style memory management with explicit read/write tools, (3) compressive transformers that maintain a compressed summary of the distant past. The tradeoff between fidelity (full buffer) and cost (summary) depends on the application. For conversational agents, a sliding window of 5–10 turns is a reasonable default.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter1_core/class2_memory_conversation/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter1_core/class2_memory_conversation/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `buffer_ok` | [1, 1] |
| `history_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
