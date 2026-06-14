# Exam Plan — sLM Universe Learning

## Why this exists
Plan written before any exam file is created or modified. Edit this file first when scope changes.

---

## Target exam set (6 notebooks)

| File | Domain | Questions | Status |
|---|---|---|---|
| `exam_01_sLM_mastery.ipynb` | General overview — all courses, all projects | 55+ | exists, incomplete |
| `exam_02_finetuning_qlora_dpo.ipynb` | Course 1 deep dive: LoRA math, QLoRA memory, DPO, contrastive FT | 50+ | TODO |
| `exam_03_continual_learning.ipynb` | Course 2: EWC, replay, LoRA-per-task, forgetting metrics | 50+ | TODO |
| `exam_04_pipelines_production.ipynb` | Course 3 + projects: deployment, Docker, GCP, FastAPI, monitoring | 50+ | TODO |
| `exam_05_langchain_agents.ipynb` | Course 4: LCEL, RAG, LangGraph, LangSmith, production patterns | 50+ | TODO |
| `exam_06_interview_simulation.ipynb` | Full Big Tech simulation: system design + live coding + debugging | 30+ | TODO |

---

## Question structure per notebook

Each notebook follows this layout:

```
## Part I  — MCQ              (20 questions, 1 pt each)
## Part II — Short Answer     (15 questions, 2 pt each, 3-5 sentences)
## Part III — Coding Challenge (10 questions, starter code cells)
## Part IV — Long Form        (5 questions, system design / case study)
```

Total per exam: ~50 questions.

### Question grounding rules
- At least **half** of MCQ and SA questions reference actual files/configs in
  `learning_slm_claude/courses/` or `courses/projects/` (specific class names,
  YAML keys, actual error messages from train logs).
- **All coding cells are fully self-contained with mocks** — no real model
  downloads, no GPU required, no external API calls, no network access:
  - Replace `AutoModelForCausalLM.from_pretrained(...)` with a tiny
    2-layer transformer or `unittest.mock.MagicMock`.
  - Replace FAISS/ChromaDB with an in-memory dict or numpy dot-product.
  - Replace LangChain LLM calls with a `FakeLLM` that returns a fixed string.
  - Replace HTTP calls (GCP, HuggingFace Hub) with `responses` mock or simple stub.
  - Replace file I/O with `tempfile.TemporaryDirectory()`.
  The rule: every code cell must run to completion with `conda activate llms`
  on a CPU-only laptop with zero internet, producing visible output.
- No generic "what is LoRA?" questions — always anchor to a specific tradeoff,
  code path, or config value from the curriculum.

---

## exam_01 — what's missing (completion plan)

Current state: 25 MCQ + 15 SA + 8 coding + 5 case studies = 53 questions.

Gaps identified:
- Part I is missing Course 2 (continual learning) and Course 3 (pipeline/drift) topics entirely.
- Part III coding challenges are not self-contained — several require imports not in starter cells.
- Part IV case studies lack grading rubric cells.
- No auto-grader for SA / CC parts (only MCQ has one).

Fixes needed:
1. Add 5 MCQ covering Course 2 (EWC, replay) and Course 3 (PSI drift, blue/green).
2. Fix each CC starter cell to include all needed imports.
3. Add a rubric markdown cell after each case study.
4. Add a SA spot-check cell (prints question numbers user filled in).

---

## exam_02 — Fine-tuning Deep Dive (outline)

Topics to cover:
- LoRA rank math (trainable params formula)
- QLoRA: NF4 vs INT8 memory trade-offs
- `prepare_model_for_kbit_training` ordering
- `device_map="auto"` vs explicit placement
- DPO loss formula; reference model role
- SFT loss vs DPO loss sign
- Contrastive embedding (MNRL, InfoNCE)
- RAG pipeline: embed → retrieve → generate
- Eval metrics: BLEU, ROUGE, exact-match, MRR
- Gotcha: `return_full_text=False` in HF pipeline

---

## exam_03 — Continual Learning (outline)

Topics to cover:
- Catastrophic forgetting: what it is, why it happens
- EWC: Fisher information matrix intuition + lambda tuning
- Experience replay: reservoir sampling, memory budget
- LoRA-per-task: shared trunk, task-specific adapters
- Metrics: BWT (backward transfer), FWT (forward transfer)
- Eval harness: `eval_harness.py` shared module role
- PSI drift detection: population stability index formula
- Blue/green deployment for model swap
- Auto-update pipeline trigger logic

---

## exam_04 — Pipelines & Production (outline)

Topics to cover:
- FastAPI lifespan context manager vs deprecated `on_event`
- `parents[N]` path resolution in Docker containers
- GCP Cloud Run: scale-to-zero, memory/CPU limits
- GitHub Actions: `needs:`, `id-token: write`, `google-github-actions/auth@v2`
- Artifact Registry: image URI format, tagging with `${{ github.sha }}`
- `HF_HOME=/tmp` for read-only Cloud Run filesystem
- Semantic cache: cosine threshold, when to skip LLM
- RAGAS: faithfulness vs context recall difference
- Docker Compose: named volumes, healthcheck syntax
- `release` branch deploy strategy; why not `main`

---

## exam_05 — LangChain & Agents (outline)

Topics to cover:
- LCEL pipe operator `|` and chain composition
- `RunnableWithMessageHistory` vs raw ConversationBufferMemory
- `PydanticOutputParser` vs `with_structured_output()`
- FAISS vs ChromaDB: persistence, search type, when to use each
- MMR retrieval: diversity vs relevance
- Multi-query retrieval: why and how
- LangGraph: StateGraph, nodes, edges, conditional routing
- ReAct loop: thought/action/observation cycle, termination
- `bind_tools()` requirement; why HuggingFacePipeline can't use it
- LangSmith: trace anatomy, offline fallback

---

## exam_06 — Full Interview Simulation (outline)

Format: simulate a real onsite. No hints, no starter code for some questions.

- Section A: 45-min coding (2 problems — medium LeetCode-style with ML twist)
- Section B: 30-min system design (design a production RAG system from scratch)
- Section C: 15-min debugging (given broken code, find and fix 3 bugs)
- Section D: 10-min ML theory (5 rapid-fire questions, no multiple choice)

---

## Execution order

1. Fix `exam_01` gaps (see above) — commit as amendment to same branch
2. Build `exam_02` (fine-tuning deep dive)
3. Build `exam_03` (continual learning)
4. Build `exam_04` (pipelines + production)
5. Build `exam_05` (LangChain + agents)
6. Build `exam_06` (interview simulation)
7. All 6 on same `feat/exams-sLM-mastery` branch → single PR

---

## Verification per exam

- Open in JupyterLab: `conda activate llms && jupyter lab exams/examNN_*.ipynb`
- Run all code cells — zero `ImportError` or `NameError`
- MCQ auto-grader cell must print a score
- At least one coding cell per exam must produce output when run
