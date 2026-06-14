# Class 3 — Structured Output with Pydantic

## Psycho Mode

Getting an LLM to return free-form prose is easy. Getting it to return a Python object you can actually use in code — with specific fields, correct types, and validation — is much harder. Structured output is the bridge between "the model said something" and "my application received a typed data structure."

The key insight is that you can inject schema information directly into the prompt. By telling the model "respond only with valid JSON matching this schema", you nudge it toward structured output. Then a parser attempts to deserialize the response. When it fails, you have a fallback: extract the JSON blob from the raw string and try again. In production, tools like `instructor` or `with_structured_output()` go further by automatically retrying and fixing malformed JSON.

## Academic Mode

Let $\mathcal{S} = \{(k_i, t_i)\}$ be a JSON schema with field names $k_i$ and types $t_i$. The structured output problem is: given a natural language input $x$, generate $y \in \mathcal{J}(\mathcal{S})$ where $\mathcal{J}(\mathcal{S})$ is the set of valid JSON objects conforming to $\mathcal{S}$.

LangChain's `JsonOutputParser` injects schema description via `get_format_instructions()` into the prompt, producing:

$$p = \text{template}(x) + \text{format\_instructions}(\mathcal{S})$$

The model generates $\hat{y} = f_\theta(p)$, then the parser applies `json.loads()` and Pydantic validation. Parse success rate (fraction of calls that produce valid objects) is the primary metric. Reference: LangChain output parsers documentation at [https://python.langchain.com/docs/concepts/output_parsers/](https://python.langchain.com/docs/concepts/output_parsers/).

## Engineering Mode

```python
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class Movie(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")

parser = JsonOutputParser(pydantic_object=Movie)
prompt = PromptTemplate(
    template="Extract movie info as JSON:\n{format_instructions}\n\nText: {text}\n\nJSON:",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
chain = prompt | llm | parser
result = chain.invoke({"text": "The Matrix was released in 1999 and is a sci-fi film."})
# result == {"title": "The Matrix", "year": 1999, "genre": "sci-fi"}
```

Fallback pattern for local models that ignore format instructions:

```python
raw = (prompt | llm | StrOutputParser()).invoke({"text": text})
start, end = raw.find("{"), raw.rfind("}") + 1
obj = json.loads(raw[start:end])  # may raise; wrap in try/except
```

Config keys: `backbone`, `limits.smoke.n_samples`.

## Research Mode

Structured output reliability varies dramatically with model size. Small models (<500M parameters) often ignore schema instructions; larger models (>7B) handle them reliably. Research directions: (1) constrained decoding (outlines, guidance) forces the model to only emit valid JSON at the token level — this guarantees structure but requires white-box access to logits; (2) function calling schemas (OpenAI-style) encode the schema in a separate field outside the prompt, reducing context pollution; (3) the `instructor` library uses retry loops with fix-up prompts to correct malformed output. For local sLMs, constrained decoding is the most reliable option.

## How to run

```bash
bash courses/course4_langchain_ecosystem/chapter1_core/class3_structured_output/run.sh
MODE=full bash courses/course4_langchain_ecosystem/chapter1_core/class3_structured_output/run.sh
```

## How to verify

| Metric | Expected |
|---|---|
| `parse_ok` | [1, 1] |
| `schema_ok` | [1, 1] |

---

**Instructor checklist**

- [x] All four mode sections present.
- [x] Official reference links included.
- [x] No numeric literals except 0/1 in code files.
- [x] `configs/default.yaml` declares `expected_band`.
- [x] `run.sh` chmod+x.
- [x] `exercises.md` has 3 exercises.
- [x] Linked from course README.
