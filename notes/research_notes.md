# Research Notes

This file grounds specific design decisions in this repo against the actual
papers on hand (full PDF text read directly, not summarized secondhand).
Each entry maps to a chapter checkbox in the main README.

---

## Papers on hand

1. Khan, A.A., Hasan, M.T., Kemell, K.K., Rasku, J., Abrahamsson, P. (2024).
   *Developing Retrieval Augmented Generation (RAG) based LLM Systems from
   PDFs: An Experience Report.* arXiv:2410.15944.
2. Zhang, D., Xu, X., Wang, C., Xing, Z., Mao, R. (2024). *A Layered
   Architecture for Developing and Enhancing Capabilities in Large Language
   Model-based Software Systems.* arXiv:2411.12357.
3. Sharma, C. (2025). *Retrieval-Augmented Generation: A Comprehensive Survey
   of Architectures, Enhancements, and Robustness Frontiers.* arXiv:2506.00054.
4. Gotavade, T.S. *Artificial Intelligence Ecosystem for Automating
   Self-Directed Teaching.* Terna Engineering College, Navi Mumbai.
5. Yu, C., Cheng, Z., Cui, H., Gao, Y., Luo, Z., Wang, Y., Zheng, H., Zhao, Y.
   (2025). *A Survey on Agent Workflow – Status and Future.* Sichuan
   University.
6. Bernardi, M.L., Casciani, A., Cimitile, M., Marrella, A. (2024).
   *Conversing with business process-aware large language models: the BPLLM
   framework.* Journal of Intelligent Information Systems, 62, 1607–1629.
7. Heiland, L., Hauser, M., Bogner, J. (2023). *Design Patterns for AI-based
   Systems: A Multivocal Literature Review and Pattern Repository.*
   arXiv:2303.13173.
8. Fadeeva, E., Rubashevskii, A., Piatrashyn, D., Vashurin, R., Dhuliawala, S.,
   Shelmanov, A., Baldwin, T., Nakov, P., Sachan, M., Panov, M. (2025).
   *Faithfulness-Aware Uncertainty Quantification for Fact-Checking the
   Output of Retrieval-Augmented Generation (FRANQ).* arXiv:2505.21072.
9. Packowski, S., Halilovic, I., Schlotfeldt, J., Smith, T. (2024).
   *Optimizing and Evaluating Enterprise Retrieval-Augmented Generation
   (RAG): A Content Design Perspective.* ICAAI '24. arXiv:2410.12812.
10. Venkatesh, K., Dalva, Y., Lourentzou, I., Yanardag, P. (2024). *Context
    Canvas: Enhancing Text-to-Image Diffusion Models with Knowledge
    Graph-Based RAG.* arXiv:2412.09614.

Note: Paper 10 (Context Canvas) is a text-to-image RAG paper, not text QA —
included because it was in the original document set, but it's tangential to
this repo's scope (Chapters 0-4 are all text-based RAG/agents). Not cited
below since none of its claims apply here.

---

## Chapter 0 — OOP Foundations / Layered Architecture

**Grounds:** why splitting the system into `BaseLLM` / `BaseEmbedder` /
`BaseVectorStore` interfaces, rather than one monolithic script, is a
recognized architectural pattern, not just a style preference.

**Design Patterns for AI-based Systems** (Heiland et al., 2023) is the direct
source for this. Their literature review identifies the **Strategy Pattern**
as an adapted traditional pattern specifically used in AI systems: "define an
interface (strategy) that different models implement. The context will call
the methods exposed by the interface, and the implemented models will behave
differently based on the contextual data." Their listed real-world example
is HuggingFace's pipeline interface — structurally identical to this repo's
`BaseEmbedder`/`BaseLLM` abstract classes with swappable concrete
implementations (`LocalEmbedder` vs `OpenAIEmbedder`, `GroqLLM`).

Their catalogue also names the **AI Pipeline** pattern (aka Sequential
Decomposition): dividing a problem into smaller consecutive steps, then
combining several existing AI tools or custom models into an inference-time
pipeline where each specialized tool or model is responsible for a single
step. This is the direct justification for `RAGPipeline` chaining embedder →
vector store → LLM as separable, swappable stages rather than one fused
function.

**A Layered Architecture for LLM-based Software Systems** (Zhang et al.,
2024) supplies the second half of the justification: organizing an LLM
system into Model Layer, Inference Layer, and Application Layer, each with
distinct components and a distinct "developer access" surface. This repo's
structure maps onto their layers directly:
- `src/embedders`, `src/llms` = swappable Model-Layer-facing components
- `src/pipeline.py`, `src/agent.py` = Application Layer (their "Mechanism
  Engineering" and "Orchestration" components)

Zhang et al.'s **Tooling** section also directly names this repo's Chapter 3
design before it existed: they distinguish "passive tool calling" (vanilla
RAG — retrieval decided externally, LLM unaware tools exist) from "active
tool calling" (LLM has agency to decide which tool to invoke). The
`RAGQueryTool`/`CalculatorTool` + `Agent.run()` design in this repo is their
active-tool-calling case.

---

## Chapter 1 — Production RAG Architecture

**Grounds:** ingestion → chunking → retrieval → generation pipeline design,
and the section-aware chunking decision specifically.

Khan et al. (2024) document the RAG pipeline as seven concrete stages: data
collection, preprocessing, vector embedding creation, retrieval, context
augmentation, LLM generation, final output — the shape `RAGPipeline` follows
directly. They also explicitly recommend **semantic chunking** over
arbitrary splitting: use semantic chunking based on logical divisions within
the text, such as paragraphs or sections, so that each chunk retains its
context. This is the direct justification for `RAGPipeline._chunk_text()`'s
header-then-paragraph fallback strategy.

**Optimizing and Evaluating Enterprise RAG** (Packowski et al., 2024, IBM) —
the paper your original README already named for Chapter 1 — adds a sharper,
more production-tested version of the same point, based on running RAG at
enterprise scale: chunking too small risks splitting information across
multiple chunks, while chunking too large risks including irrelevant
information; one way to include a complete, self-contained idea in each
chunk is to chunk at the chapter or section level rather than by size. Their
team's strategy — extracting whole "topics" (their term for sections)
instead of fixed-size chunks, which they call "small2big" or "parent
document retrieval" — is functionally the same idea this repo's
section-aware chunker implements, validated against real user traffic rather
than a benchmark.

Packowski et al. also report something worth carrying into this repo's
future work: their team's biggest accuracy gains came not from a smarter
retriever or a bigger model, but from **rewriting the knowledge base content
itself** to be easier to retrieve and quote from — e.g. one single missing
number in a source paragraph caused a wrong answer, fixed by adding that one
number back in. This complicates a pure "better chunking" framing: sometimes
the fix isn't the pipeline, it's the source documents.

BPLLM (Bernardi et al., 2024) provides a directly comparable **empirical
chunking benchmark** worth noting for future chunking work in this repo:
testing fixed-size vs. recursive vs. format-specific chunking on a business
process model, their format-specific chunker (analogous in spirit to this
repo's header-aware chunker) reached 76.47% accuracy vs. 60.57% for no
chunking, on the same LLM. This is a concrete number showing format-aware
chunking isn't just intuitively better — it measured 16 points higher in
their setup.

Sharma (2025)'s survey gives the formal vocabulary and mathematical
definition for what this pipeline is: retrieval-weighted conditional
generation over top-k documents (their Eq. 1–2) — the retriever-centric,
generator-passive design in their taxonomy (Section 3.1), where the retriever
does the work of finding relevant chunks and the generator is a comparatively
passive decoder conditioned on that context.

---

## Chapter 2 — Evaluation & Guardrails (not yet built)

**Grounds for the next implementation step.**

**FRANQ** (Fadeeva et al., 2025) is the paper your original README already
named for this chapter, and it directly reframes what Chapter 2's checklist
item ("does the generated answer's content actually appear in the retrieved
context?") should actually measure. FRANQ's key argument: **faithfulness and
factuality are not the same thing**, and conflating them is a common
mistake — a claim may be factually true yet not grounded in the retrieved
context, or faithful to the retrieval while still being factually incorrect.
A simple "does the answer appear in context" check (which is what Chapter
2's original checklist implies) is actually a **faithfulness** check, not a
factuality check — it would incorrectly flag a true-but-unretrieved fact as
a hallucination, and would pass a retrieved-but-wrong fact as fine.

For a build this size, FRANQ's full pipeline (claim decomposition + trained
uncertainty classifiers + isotonic calibration) is heavier than needed, but
the core idea is directly implementable as a lightweight two-part check on
top of the existing `RAGPipeline`:
1. **Faithfulness check** — does the retrieved context actually entail the
   generated claim? (FRANQ uses AlignScore, a trained model; a cheaper
   version for this repo's scale could just ask the LLM itself "is this
   claim entailed by this context, yes/no".)
2. **Out-of-context flag, not automatic rejection** — if a claim isn't
   faithful, that alone doesn't mean it's wrong (per FRANQ's core argument);
   it just means it needs a different kind of check before trusting it.

Packowski et al. (2024) independently confirm, from real production RAG
operation, that common benchmark evaluation techniques were not useful for
evaluating responses to novel user questions, so their team relies on a
"human in the lead" review tool tracking specific failure categories instead
(no matching content exists, search failed to find it, LLM answer was poor).
This is a useful, more practical alternative shape for Chapter 2 if a full
FRANQ-style classifier is more than this project needs: a small
failure-category tracker rather than a scored classifier.

Gotavade's paper reports simpler, directly implementable metrics used in a
working RAG system: ROUGE-1/2/L, average BLEU, cosine similarity between
generated and reference answers, and a "hallucination rate" defined as the
percentage of generated responses staying relevant to retrieved documents
(reported at 0.99 in their system). These are a lower-effort fallback if
FRANQ's approach is more than this project needs.

---

## Chapter 3 — Autonomous Agents (built)

**Grounds:** the ReAct-style bounded loop, tool routing, and the
decide/act separation already implemented in `src/agent.py`.

**A Survey on Agent Workflow** (Yu et al., 2025) is the paper your original
README named for this chapter, and it directly names and formalizes what
`Agent.run()` already does:

- Their core agent definition matches this repo's design exactly: agents are
  systems where LLMs dynamically direct their own processes and tool usage,
  maintaining control over how they accomplish tasks.
- They describe the **ReAct pattern** by name as the dominant approach: an
  agent that interleaves natural language reasoning traces with
  task-specific actions, enabling LLMs to update plans, handle exceptions,
  and interact with external environments in a structured loop. This is
  exactly `Agent.run()`'s `Action: tool | input` / `Final Answer:` loop.
- Their description of LangChain's agent loop is essentially a spec for this
  repo's implementation: the LLM reads the request, decides whether to
  answer directly or use a tool, the framework runs the tool and gives the
  result back to the LLM, and this back-and-forth continues until the LLM
  gives a final answer.
- Critically, the checklist item most relevant to Chapter 3 — **bounded,
  controllable behavior** — is something they flag as a known gap across the
  field, not a solved problem: their Limitations section explicitly lists a
  lack of unified evaluation metrics for agents, and notes most systems
  don't have hard caps and can loop indefinitely. `Agent`'s `max_steps` cap
  directly addresses this named gap rather than assuming it away.
- They also describe ReWoo as a contrasting alternative worth knowing about:
  a "reasoning-without-observation" pattern that plans all tool calls
  upfront and executes them together, rather than interleaving reasoning and
  action step-by-step like ReAct. Worth considering later if `Agent`'s
  per-step LLM calls become a latency/cost concern — ReWoo trades adaptivity
  for fewer LLM calls.

**A Layered Architecture** (Zhang et al., 2024, cited above under Chapter 0)
supplies the "clear handoff between decide and act" framing already used in
`Agent`'s docstring: the LLM only ever produces text describing an action;
`Agent` — not the LLM — is the only thing that actually executes
`tool.run()`.

**BPLLM** (Bernardi et al., 2024) is a good worked precedent for combining
RAG with a narrower, task-specific tool (in their case, business process
querying) rather than a general chatbot — structurally close to how
`RAGQueryTool` wraps this repo's Chapter 1 pipeline as one tool among
several, rather than being the whole system.

---

## Chapter 4 — Capstone (not yet built)

Gotavade's paper is a full worked example of Chapter 4's ask — one concrete
educational workflow built by combining RAG, fine-tuning, and an agent-like
virtual tutor loop, including a real decision framework for RAG vs.
fine-tuning vs. RAFT (retrieval-aware fine-tuning) trade-offs. Worth
revisiting once Chapter 4 is scoped.

BPLLM (Bernardi et al., 2024) is also directly relevant here as a second
worked capstone example: "answer questions about a specific business
process/workflow from natural language" is close in shape to your README's
own suggested Chapter 4 example ("answer internal policy questions from
company docs"), and BPLLM's paper reports real accuracy numbers (RQ1-RQ4) for
exactly that kind of use case, which is useful for setting realistic
expectations before building.

Packowski et al. (2024)'s enterprise deployment experience is also relevant
to Chapter 4 specifically because it's the most production-tested of all
these papers: their content-design guidelines (avoid deeply nested lists,
add summaries to long procedures, explain graphics in text, avoid complex
tables) are concrete, actionable advice for whatever documents end up in the
Chapter 4 capstone's knowledge base — worth applying to `data/sample_docs/`
content design directly, not just as a citation.
