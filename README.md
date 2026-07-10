# Production-Ready RAG & Autonomous Agents — A Chapter-Based Training Path

> Object-oriented training for independently designing and building AI systems and
> autonomous agents that solve business problems and automate workflows.

This repo is structured so a learner can go chapter by chapter, building a real,
working, production-minded RAG + agent system from scratch — not just reading
about one.

---

## How to use this repo

Each chapter covers a concept, the research grounding behind it, what was built
hands-on, and how it holds up against real production concerns. Work through
chapters in order — each one builds on the last.

---

## Chapter 0 — OOP Foundations for AI Systems

**Concept:** Why production AI code is built on abstract interfaces, not scripts.
Classes, abstract base classes (ABC), inheritance, composition, and why
interface-driven design lets you swap LLM/embedding/vector-store providers
without rewriting your pipeline.

**Research grounding:** (see `notes/research_notes.md` for full summaries)

- _Design Patterns for AI-based Systems_ — Heiland et al. (2023)
- _A Layered Architecture for LLM-based Software Systems_ — (2024)

**What was built:** `BaseLLM`, `BaseEmbedder`, and `BaseVectorStore` as abstract
classes, each with concrete implementations — local embeddings + Groq are the
active providers, and OpenAI-backed versions of both are implemented but unused,
specifically to prove the interface actually swaps cleanly. Every class is typed
and documented with docstrings.

**Design decision:** local embeddings (`sentence-transformers`,
`all-MiniLM-L6-v2`) and Groq's hosted LLM API (`llama-3.1-8b-instant`) were used
instead of OpenAI, primarily due to cost constraints during development.
Because `BaseEmbedder`/`BaseLLM` are abstract interfaces, moving to OpenAI or
another provider in production requires changing only the concrete
implementation class — `pipeline.py` itself needs zero changes. `OpenAIEmbedder`
is kept in the repo, implemented but unused, as a direct demonstration of that
swap.

---

## Chapter 1 — Production RAG Architecture

**Concept:** The real pipeline — ingestion, chunking, embedding, retrieval,
generation — and the design decisions that separate a demo from something that
survives real documents and real traffic.

**Research grounding:**

- _Optimizing and Evaluating Enterprise RAG_ — Packowski et al. (2024)
- Retriever/Generator/Hybrid RAG taxonomy paper

**What was built:** `RAGPipeline` ties embedder, vector store, an optional
reranker, and the LLM together. Ingestion is multi-format — PDF, DOCX, PPTX,
HTML, and TXT (`src/ingestion/extractors.py`) — reading real documents rather
than hardcoded strings. Chunking is section-aware: it splits on markdown-style
headers first, falls back to paragraph-level splitting for any oversized
section, strips a standalone leading title before chunking, and drops
sections below a minimum length so a bare heading with no real content
underneath doesn't pollute retrieval. An optional reranking stage
(`FlashRankReranker`) narrows a wider retrieval pool down to the most relevant
chunks before they reach the LLM. Every retrieved chunk also carries
`{"source": filename}` metadata, so an answer can always be traced back to
which document it came from — surfaced directly in the Streamlit UI.

Ingested across five file formats, the pipeline currently holds 34 chunks from
a small realistic document set. `GroqLLM.generate()` wraps its API call and
raises a `RuntimeError` with context on failure, rather than failing silently
or returning garbage.

### Validation

1. **Grounded answer check** — asked questions with specific answers present
   in the ingested documents (e.g. API rate limits, lodging expense caps).
   Correctly answered with the exact figures from the source documents.

2. **Out-of-scope / hallucination check** — asked a question with no answer in
   any ingested document. Correctly declined rather than fabricating an
   answer, confirming the "answer only from context" guardrail in the prompt
   works.

3. **Cross-document discrimination check** — asked questions answerable only
   from specific documents among the five ingested (policy doc vs. FAQ vs.
   slide deck vs. technical guide). Correctly retrieved from the right
   document each time, rather than defaulting to whichever document was
   ingested first.

4. **Chunking bug found and fixed** — an early version let short,
   heading-only chunks (e.g. a title slide with no body text) into the
   vector store, where their short, generic embeddings sometimes outranked
   real content. Fixed by raising the minimum chunk length threshold and
   stripping standalone leading titles before chunking (`_strip_leading_title`
   in `src/pipeline.py`).

---

## Chapter 2 — Evaluation & Guardrails

**Concept:** How to know if your RAG system is actually telling the truth.
Faithfulness, hallucination detection, uncertainty quantification.

**Research grounding:**

- FRANQ (Faithfulness-aware RAG Uncertainty Quantification) paper

**What was built:** Faithfulness and answer-relevancy scoring via RAGAS
(`test_evaluation.py`), with every run's question, answer, retrieved context,
and scores logged to `evaluation_log.csv` — so a bad answer can be traced back
to why it happened, and there's a concrete signal available before an answer
would ever reach a user.

**Note on FRANQ vs. this implementation:** FRANQ's core insight — that
faithfulness (does the answer match retrieved context) and factuality (is the
answer actually true) are distinct and require different detection strategies
— directly shaped how this evaluation is scoped. RAGAS's faithfulness metric
checks the former, not the latter; a fully faithful answer can still be wrong
if the retrieved context itself is wrong. This is flagged in
`notes/research_notes.md` as a known limitation, not solved here.

---

## Chapter 3 — Autonomous Agents & Workflow Automation

**Concept:** Moving from "retrieve then generate" to agents that reason, use
tools, and take multi-step action — the ReAct pattern, bounded reasoning
loops, and a clear separation between the LLM deciding and the system acting.

**Research grounding:**

- Agent workflow survey (2025)
- BPLLM — business-process-aware LLMs

**What was built:** A tool-using `Agent` class (`src/agent.py`) on top of the
Chapter 1 pipeline, with a `rag_query` tool and a `calculator` tool, tested in
`test_agent.py`. The reasoning loop is capped by `max_steps`, so a confused
LLM can't loop forever — the LLM only ever produces `Action:` / `Final
Answer:` text, and the `Agent` class is the only thing that actually executes
`tool.run()`. That separation is deliberate: the model decides, the code
acts.

---

## Chapter 4 — Capstone: New Employee Onboarding Assistant

### The business problem

A new AICertify employee has questions during onboarding — AI ethics policy,
travel/reimbursement limits, deployment procedures — that would normally mean
pinging HR or a compliance officer and waiting for a reply. This capstone
automates that first line of response: an assistant that answers onboarding
questions directly from the company's actual documents, including questions
that require doing a calculation on top of a looked-up fact (e.g. "the limit
is X, I've used Y, how much do I have left").

### How Chapters 0-3 combine to solve it

**Chapter 0 (OOP foundations)** is what makes the other three chapters
composable at all. `RAGPipeline` and `Agent` are built against abstract
interfaces (`BaseEmbedder`, `BaseLLM`, `BaseVectorStore`, `BaseReranker`,
`BaseTool`), not concrete providers. `onboarding_assistant.py` plugs in
`LocalEmbedder`, `GroqLLM`, `ChromaVectorStore`, and `FlashRankReranker`
without either `RAGPipeline` or `Agent` needing to know or care — the same
code would work with different providers swapped in underneath.

**Chapter 1 (Production RAG)** supplies the actual knowledge base this
assistant answers from: multi-format ingestion covers the realistic mix of
document types a real onboarding packet would contain, section-aware
chunking keeps each retrieved chunk semantically whole instead of cutting
policy language mid-sentence, and reranking narrows a wider candidate pool
down to the most relevant chunks before they ever reach the LLM.

**Chapter 2 (Evaluation)** is what stands between "the pipeline runs" and
"this is safe to hand a new employee." `test_evaluation.py`'s RAGAS
faithfulness/relevancy scoring is exactly the gate that should run on this
assistant's outputs before trusting them unsupervised — a new hire acting on
a hallucinated reimbursement figure is a real, undesirable outcome.
Evaluation isn't run inline in `onboarding_assistant.py`, but it's the same
`RAGPipeline` object Chapter 2 scores.

**Chapter 3 (Agent)** is the piece that turns this from "a search box" into
"an assistant that reasons." The onboarding questions here are deliberately
mixed: some need only a policy lookup (`rag_query`), but one needs a policy
lookup _and_ a calculation chained together — the agent has to decide, across
its own bounded reasoning steps, to call `rag_query` first to find the
reimbursement limit, then call `calculator` to subtract the amount already
claimed. A plain RAG pipeline (Chapter 1 alone) cannot do this reliably — it
can retrieve the limit, but it can't be trusted to do arithmetic on top of
what it retrieves. This is the concrete case where Chapter 3's tool-routing
agent earns its place over Chapter 1 alone.

### What this demonstrates

Given a new-hire-facing question that spans a knowledge lookup and a
calculation, the system retrieves the relevant policy section from the
correct source document among five different formats (Chapter 1), would be
checked for faithfulness before being trusted in production (Chapter 2),
autonomously decides it needs a second tool call rather than stopping at the
retrieved fact and completes the calculation itself (Chapter 3) — all running
through swappable, interface-based components that could be redeployed with
different providers without touching this script (Chapter 0).

---

## Repo structure

```
rag-project/
├── src/
│   ├── embedders/
│   │   ├── base_embedder.py       (Ch 0 — abstract interface)
│   │   ├── local_embedder.py      (Ch 0 — active, sentence-transformers)
│   │   └── openai_embedder.py     (Ch 0 — interface-complete, unused)
│   ├── llms/
│   │   ├── base_llm.py            (Ch 0 — abstract interface)
│   │   ├── groq_llm.py            (Ch 0 — active)
│   │   └── openai_llm.py          (Ch 0 — interface-complete, unused)
│   ├── vectorstores/
│   │   ├── base_vectorstore.py    (Ch 0 — abstract interface, metadata-aware)
│   │   └── chroma_vectorstore.py  (Ch 0 — active, in-memory Chroma)
│   ├── loaders/
│   │   ├── document_loader.py
│   ├── rerankers/
│   │   ├── base_reranker.py       (Ch 1 — abstract interface)
│   │   └── flashrank_reranker.py  (Ch 1 — active)
│   ├── ingestion/
│   │   └── extractors.py          (Ch 1 — PDF/DOCX/PPTX/HTML/TXT extraction)
│   ├── tools/
│   │   └── calculator_tool.py
│   │   └── rag_query.py
│   │   └── base_tool.py           (Ch 3 — abstract tool interface)
│   ├── pipeline.py                (Ch 1 — RAGPipeline: ingest + chunk + retrieve + rerank + generate)
│   └── agent.py                   (Ch 3 — bounded ReAct agent)
├── data/sample_docs/true_data/    (Ch 1 — multi-format realistic documents)
├── data/sample_docs/noisy_data/   (Ch 1 — robustness/noise testing)
├── app.py                         (Streamlit chat UI — source citations shown per answer)
├── onboarding_assistant.py        (Ch 4 — capstone: combines all chapters)
├── evaluation_log.csv             (Ch 2 — logged RAGAS evaluation runs)
├── test_pipeline.py               (Ch 1 — end-to-end + validation checks)
├── test_reranker.py               (Ch 1 — reranker behavior)
├── test_rerank_ordering.py        (Ch 1 — reranking on ambiguous queries)
├── test_evaluation.py             (Ch 2 — RAGAS evaluation harness)
├── test_agent.py                  (Ch 3 — agent tool-routing test)
├── notes/research_notes.md        (research paper summaries, in own words)
├── .env                           (API keys — not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup & how to run

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here   # optional — only needed to exercise the unused OpenAI provider classes
```

(Free Groq key from https://console.groq.com/keys)

Run the Streamlit demo:

```bash
streamlit run app.py
```

Run individual test/validation scripts:

```bash
python test_pipeline.py
python test_reranker.py
python test_evaluation.py
python test_agent.py
python onboarding_assistant.py
```

## Status

Chapters 0-4 are implemented and validated, not just written. Chapter 0
provides swappable interfaces for every component. Chapter 1's pipeline
handles five file formats with section-aware chunking, retrieval, and
reranking, validated with grounded/out-of-scope/cross-document checks.
Chapter 2 scores real outputs with RAGAS and logs them for traceability.
Chapter 3's agent is bounded and tested. Chapter 4 ties all four together
into one concrete business workflow. Research grounding for each chapter is
documented, in original wording, in `notes/research_notes.md`.

### Honest limitations / next steps

The vector store (Chroma) runs in-memory and resets on restart; a production
deployment would need a persistent store. There's no production
observability beyond the evaluation CSV — a real deployment would add
structured logging and monitoring. The agent implemented here is
single-agent, not multi-agent; the agent workflow survey in the research
notes covers multi-agent orchestration patterns that were scoped out given
the project timeline. There's also no guardrails layer for adversarial or
malicious input, which the RAG security/robustness literature in the
research notes flags as a real production requirement not yet addressed
here.
