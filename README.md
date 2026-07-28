# 🌽 Crop Advisory ReAct Agent

**Author:** Linlin Zhang · [github.com/linz21](https://github.com/linz21)

A ReAct-style agent that helps corn farmers by combining two tools built in
earlier projects: a yield prediction model (Project 1) and an agronomic
research literature search system (Project 3). The agent reasons about
which tool(s) a question needs, calls them, and synthesizes a final answer,
citing real sources.

## Architecture

```
User question
        ↓
Hand-rolled ReAct loop (local Qwen3-4B, or Anthropic API — see Results)
        ↓                                    ↓
predict_corn_yield tool              search_literature tool
(calls Project 1's live API)         (calls Project 3's retriever +
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
model used successfully in Project 3), but empirical testing showed a real
synthesis failure — correct tool selection, but vague answers even given
good, specific tool results. Upgraded to `Qwen3-4B-Instruct-2507`, a newer
generation with reported improvements in reasoning and tool use, which
measurably improved synthesis quality. A further limitation found in both
local models (see Results) motivated adding the Anthropic API as an
optional, config-switchable alternative (`llm.provider: "anthropic"`).

## Setup

This project depends on both earlier projects being available:

```bash
# 1. Clone this repo and Project 3 as SIBLING directories
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

# 5. (Optional) Use Anthropic API instead of local model — see Results
#    for why this is offered as an alternative. Requires a genuine API
#    key from console.anthropic.com with billing enabled, NOT a Claude
#    Pro/Max subscription (separate product/billing).
export ANTHROPIC_API_KEY="sk-ant-..."
# then set llm.provider: "anthropic" in configs/config.yaml

# 6. Run the agent
python -m src.agent.react_agent --question "What corn yield should I expect in Illinois next year?"
```

> **Note:** Project 1's yield prediction tool calls a live AWS EC2 endpoint,
> which may be paused to manage cloud costs (documented in that project's
> README). The tool handles this gracefully and reports the service as
> temporarily unavailable rather than crashing.
>
> **Shared environment constraint:** this project and Project 3 run in the
> same Python environment (Project 3's retriever/generator are imported
> in-process, not called over HTTP). Their dependency versions must stay
> mutually compatible — currently `torch>=2.6,<2.8` and
> `transformers>=4.51,<5.0` satisfy both projects' requirements.

## Tech Stack

Hand-rolled `ReAct` loop · `transformers` (Qwen3-4B-Instruct-2507, local) ·
optional Anthropic API (Claude Sonnet 4.5) · `Redis` (short-term memory) ·
`ChromaDB` + `sentence-transformers` (long-term memory) · `LangSmith`
(observability) · custom guardrails validation

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

**Honest summary:** two reasonable prompt-engineering attempts (different
wording, different position in the prompt) did not fix either local-model
failure. Switching to Claude Sonnet 4.5 (same code, same prompts, just a
different `llm.provider`) resolved both immediately. This suggests the
failures are a genuine model-capacity limit, not a prompting problem —
and that for an agent whose answers could inform real farming decisions,
reliable self-correction may be worth the added API cost over a fully
free local model. Local models remain the default given cost, with the
Anthropic path available as a tested, working alternative.

**Not yet tested:** LangSmith trace inspection, Guardrails validation
against real (not synthetic) agent output, out-of-scope question handling.

## Project Structure

```
crop_advisory_agent/
├── src/
│   ├── agent/react_agent.py           # Hand-rolled ReAct loop + local LLM wrapper
│   ├── tools/
│   │   ├── yield_prediction_tool.py   # Calls Project 1's live API
│   │   └── literature_search_tool.py  # Calls Project 3's retriever, in-process
│   ├── memory/
│   │   ├── redis_memory.py            # Short-term: same-session history via Redis Cloud
│   │   └── long_term_memory.py        # Long-term: cross-session semantic search via vector store
│   ├── observability/langsmith_setup.py  # LangSmith tracing configuration
│   └── guardrails/validators.py       # Output validation (regex-based starting point)
├── tests/test_pipeline.py             # Unit tests, including regression tests for
│                                       # real bugs found (e.g. source-citation parsing)
├── configs/config.yaml                # All settings — single source of truth
└── .github/workflows/ci.yml           # Tests on every push
```
