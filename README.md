# 🌽 Crop Advisory ReAct Agent

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A ReAct-style agent that helps corn farmers by combining two tools built in
earlier projects: a yield prediction model (Corn Yield Prediction) and an agronomic
research literature search system (Agricultural RAG System). The agent reasons about
which tool(s) a question needs, calls them, and synthesizes a final answer,
citing real sources.

## Live Demo

**[crop-advisory-agent.streamlit.app](https://crop-advisory-agent.streamlit.app/)**

This is a deliberately **simplified** public deployment — see
`streamlit_deploy/` and its own section below for what's different from
the full local version.

## Architecture (full local version)

```
User question
        ↓
Hand-rolled ReAct loop (Claude Sonnet 4.5 default, or local Qwen3-4B — see Results)
        ↓                                    ↓
predict_corn_yield tool              search_literature tool
(calls Corn Yield Prediction's live API)         (calls Agricultural RAG System's retriever +
        ↓                             generator, in-process)
        └──────────────┬──────────────────────┘
                        ↓
       Code-guaranteed source citation list
                        ↓
              Final answer to user

Redis (short-term, same-session memory)
Vector store (long-term, cross-session semantic memory)
LangSmith (reasoning trace observability)
```

**Note on the ReAct implementation:** this uses a hand-rolled, classic
text-based ReAct loop (Thought/Action/Action Input/Observation, parsed with
regex), not LangGraph's prebuilt `create_react_agent`. That helper requires
a model with native structured tool-calling (`.bind_tools()`), which a raw
local HuggingFace pipeline doesn't implement. The hand-rolled version works
with any instruction-following model and gives full visibility into the
model's raw reasoning — which proved essential for debugging.

**Note on the model:** originally built with `Qwen2.5-1.5B-Instruct` (same
model used successfully in Agricultural RAG System), but empirical testing showed a real
synthesis failure — correct tool selection, but vague answers even given
good, specific tool results. Upgraded to `Qwen3-4B-Instruct-2507`, a newer
generation with reported improvements in reasoning and tool use, which
measurably improved synthesis quality. Further testing found a real
reliability gap in both local models (see Results) that motivated adding
the Anthropic API — now the default provider (`llm.provider: "anthropic"`),
with `local` remaining fully supported for free/lower-stakes use.

## Setup

This project depends on both earlier projects being available:

```bash
# 1. Clone this repo and Agricultural RAG System as SIBLING directories
git clone https://github.com/linz21/crop_advisory_agent.git
git clone https://github.com/linz21/agri_rag_literature_ga.git
cd crop_advisory_agent

# 2. Install dependencies (both projects' requirements)
pip install -r requirements.txt
pip install -r ../agri_rag_literature_ga/requirements.txt

# 3. Set up Redis Cloud (free tier, no credit card — redis.io/cloud)
export REDIS_HOST="your-redis-host"
export REDIS_PORT="your-redis-port"
export REDIS_PASSWORD="your-redis-password"

# 4. Set up LangSmith (free tier, no credit card — smith.langchain.com)
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your-langsmith-key"
export LANGCHAIN_PROJECT="crop-advisory-agent"

# 5. Set up Anthropic API (DEFAULT provider — see Results for why).
#    Requires a genuine API key from console.anthropic.com with billing
#    enabled, NOT a Claude Pro/Max subscription (separate product/billing).
export ANTHROPIC_API_KEY="sk-ant-..."

# 6. (Optional) To use the free local model instead, set
#    llm.provider: "local" in configs/config.yaml — no API key needed,
#    but see Results for known reliability limitations.

# 7. Run the agent
python -m src.agent.react_agent --question "What corn yield should I expect in Illinois next year?"

# 8. (Optional) Run the full web UI instead of the CLI
uvicorn src.api.main:app --reload --port 8003   # Terminal 1
streamlit run src/frontend/streamlit_app.py     # Terminal 2, opens browser
```

> **Note:** Corn Yield Prediction's yield prediction tool calls a live AWS EC2 endpoint,
> which may be paused to manage cloud costs (documented in that project's
> README). The tool handles this gracefully and reports the service as
> temporarily unavailable rather than crashing.
>
> **Shared environment constraint:** this project and Agricultural RAG System run in the
> same Python environment (Agricultural RAG System's retriever/generator are imported
> in-process, not called over HTTP). Their dependency versions must stay
> mutually compatible — currently `torch>=2.6,<2.8` and
> `transformers>=4.51,<5.0` satisfy both projects' requirements.

## Tech Stack

Hand-rolled `ReAct` loop · Claude Sonnet 4.5 (default) or `transformers`
(Qwen3-4B-Instruct-2507, local, optional) · `FastAPI` (local serving) ·
`Streamlit` (chat UI, local + Streamlit Cloud) · `Redis` (short-term
memory) · `ChromaDB` + `sentence-transformers` (long-term memory) ·
`LangSmith` (observability, manually instrumented — see Results) · real
`guardrails-ai` validation · `gradio_client` (deployment only — calls
Agricultural RAG System's live Space)

## Results

Validated through direct testing (single-tool, multi-tool, and multi-turn
questions, verbose transcript inspection):

| Aspect | Finding |
|--------|---------|
| Tool selection | Correct in every test |
| Multi-tool chaining | Works — calls both tools in sequence when needed |
| Citation reliability | Not trusted to the model — sources are extracted programmatically from tool output and code-guaranteed in the final answer |
| Memory infrastructure | Both short-term (Redis) and long-term (vector store) confirmed working end-to-end — data saves and loads correctly |
| Memory *usage* (local models) | Failed — given real prior context showing no practices were mentioned, both Qwen2.5-1.5B and Qwen3-4B fabricated a false claim that practices had been discussed, and proceeded from that false premise |
| Tool-failure handling (local models) | Failed — a genuine tool error led to a confidently fabricated answer with an invented citation, instead of reporting the failure |
| Memory usage + tool-failure handling (Claude Sonnet 4.5) | Succeeded on both, across 3 independent test cases |
| Repeated "Final Answer:" blocks (local models) | Found via real Streamlit UI testing — the local model repeated the same answer 5x in a row with no Thought:/Question: marker between repeats, which an earlier version of the answer-parsing regex didn't catch. Fixed (see `_parse_final_answer`'s docstring); not further debugged beyond this fix, given the decision below. |

**Summary:** two reasonable prompt-engineering attempts (different
wording, different position in the prompt) did not fix either local-model
fabrication failure. Switching to Claude Sonnet 4.5 (same code, same
prompts, just a different `llm.provider`) resolved both immediately, and
performed well in live UI testing. This suggests the failures are a
genuine model-capacity limit, not a prompting problem. Combined with the
repetition bug above, **Claude Sonnet 4.5 is now the default provider**
(`llm.provider: "anthropic"`) despite its cost — reliable self-correction
matters for an agent whose answers could inform real farming decisions.
`local` remains fully supported and free for lower-stakes use or further
experimentation, but further local-model-specific debugging wasn't
pursued past the fixes already made, given this decision.

**Note on environment variables:** `REDIS_HOST`/`PORT`/`PASSWORD` were
added to `~/.zshrc` and persist automatically in new terminals.
`LANGCHAIN_TRACING_V2`/`LANGCHAIN_API_KEY`/`LANGCHAIN_PROJECT` and
`ANTHROPIC_API_KEY` were only set via `export` and do **not** persist —
re-export these in any new terminal before running `uvicorn`.

**Out-of-scope handling:** tested directly — a clearly unrelated question
("What's the best way to train a dog?") was correctly declined without
calling either tool, with a polite, on-brand redirect rather than a bare
refusal or a guess.

**Guardrails on real output:** confirmed running on every real agent
request via the audit log (not silently skipped), across both providers
and multiple question types. No real interaction has triggered a flag
yet — validated so far against hand-written synthetic triggers in
`tests/`, not yet against a genuine false/true positive in the wild.

## Streamlit Cloud Deployment (`streamlit_deploy/`)

A separate, self-contained deployment package — same lesson learned
deploying Agricultural RAG System to Hugging Face Spaces: a hosted single-process
environment can't run two local services (FastAPI + Streamlit) the way
local dev does, so this is one Streamlit app calling the agent directly
in-process.

**Deliberately simplified vs. the full local version:**
- **Anthropic only** — no local model option (not feasible on free-tier
  hosting; also the more reliable choice per Results above)
- **No Redis, no LangSmith, no Guardrails** — kept out to minimize the
  dependency footprint and keep the deployment simple to debug
- **Literature search calls Agricultural RAG System's already-deployed HF Space**
  directly via `gradio_client`, rather than bundling Agricultural RAG System's entire
  RAG stack (models, Chroma index, embeddings) into this deployment too

**Real issues found and fixed getting this actually working:**
- `gradio_client` resolved to an old, incompatible version (1.3.0) by
  default — the Space runs Gradio 5.31.0, causing a real "Could not fetch
  api info: Not Found" error. Fixed by pinning `gradio_client==2.6.0`.
- `Client()`'s auth parameter is `token=`, not `hf_token=`, despite the
  latter seeming more descriptive — a real, easy mistake caught by testing.
- Anonymous `gradio_client` connections get a much lower ZeroGPU quota on
  Agricultural RAG System's Space — a real test hit "You have exceeded your ZeroGPU
  quota" after only a couple of calls. Fixed by authenticating with an
  HF token for a higher quota tier.
- The remote Space's citation format (`**Sources:**` + numbered list)
  differs from the local in-process tool's format (comma-separated) —
  the source-extraction regex needed to handle both, found via a real
  malformed-output case during testing.
- `anthropic==0.34.0` hardcodes an `httpx` argument (`proxies`) that
  `httpx` 0.28+ removed entirely, crashing the deployed app with a
  `TypeError` at startup — a well-documented, common SDK/dependency
  version conflict. Fixed by unpinning `anthropic` to let pip resolve a
  current, compatible release.

See `streamlit_deploy/DEPLOY.md` for full setup steps.

## Project Structure

```
crop_advisory_agent/
├── src/
│   ├── agent/react_agent.py           # Hand-rolled ReAct loop; Anthropic (default) or local LLM
│   ├── tools/
│   │   ├── yield_prediction_tool.py   # Calls Corn Yield Prediction's live API
│   │   └── literature_search_tool.py  # Calls Agricultural RAG System's retriever, in-process
│   ├── memory/
│   │   ├── redis_memory.py            # Short-term: same-session history via Redis Cloud
│   │   └── long_term_memory.py        # Long-term: cross-session semantic search via vector store
│   ├── observability/langsmith_setup.py  # LangSmith config check
│   ├── guardrails/
│   │   ├── validators.py              # Real guardrails-ai custom validators (not a placeholder)
│   │   └── audit_log.py               # Structured JSONL logging of every interaction
│   ├── api/main.py                    # FastAPI serving layer
│   └── frontend/streamlit_app.py      # Streamlit chat UI (calls the FastAPI backend)
├── tests/test_pipeline.py             # Unit tests, including regression tests for
│                                       # real bugs found (e.g. source-citation parsing,
│                                       # repeated Final Answer blocks)
├── streamlit_deploy/                  # Self-contained Streamlit Cloud deployment —
│   ├── app.py                         # SEPARATE from the files above, not imported
│   ├── agent.py                       # from them (see Streamlit Cloud Deployment
│   ├── tools.py                       # section above for why). Anthropic-only,
│   ├── literature_tool_remote.py      # calls Agricultural RAG System's live Space via gradio_client
│   ├── requirements.txt               # instead of running its RAG stack in-process.
│   └── DEPLOY.md
├── configs/config.yaml                # All settings — single source of truth
└── .github/workflows/ci.yml           # Tests on every push
```
