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
Hand-rolled ReAct loop (local Qwen3-4B-Instruct-2507)
        ↓                                    ↓
predict_corn_yield tool              search_literature tool
(calls Project 1's live API)         (calls Project 3's retriever +
        ↓                             generator, in-process)
        └──────────────┬──────────────────────┘
                        ↓
       Code-guaranteed source citation list
                        ↓
              Final answer to user

Redis (conversation memory)  ·  LangSmith (reasoning trace observability)
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
measurably improved synthesis quality — see Results below.

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

# 5. Run the agent
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
`Redis` (conversation memory) · `LangSmith` (observability) ·
custom guardrails validation

## Results

Validated through direct testing (single-tool and multi-tool questions,
verbose transcript inspection):

| Aspect | Finding |
|--------|---------|
| Tool selection | Correct in every test — distinguishes yield questions from research questions reliably |
| Multi-tool chaining | Works — successfully calls both tools in sequence for questions needing both |
| Synthesis quality (Qwen3-4B) | Generally good; consistently and correctly flagged when retrieved sources didn't precisely match a question's scope (e.g. sweet corn vs. field corn, Illinois-specific vs. general) across multiple independent tests |
| Synthesis quality (Qwen2.5-1.5B, earlier) | Correct tool selection, but vague/generic answers even given detailed, relevant tool results — motivated the model upgrade |
| Citation reliability | Not fully trusted to the model — sources are extracted programmatically from tool output and code-guaranteed in the final answer, since the model was observed to sometimes drop real citations in favor of a vague placeholder |


## Project Structure

```
crop_advisory_agent/
├── src/
│   ├── agent/react_agent.py           # Hand-rolled ReAct loop + local LLM wrapper
│   ├── tools/
│   │   ├── yield_prediction_tool.py   # Calls Project 1's live API
│   │   └── literature_search_tool.py  # Calls Project 3's retriever, in-process
│   ├── memory/redis_memory.py         # Conversation history via Redis Cloud
│   ├── observability/langsmith_setup.py  # LangSmith tracing configuration
│   └── guardrails/validators.py       # Output validation (regex-based starting point)
├── tests/test_pipeline.py             # Unit tests, including regression tests for
│                                       # real bugs found (e.g. source-citation parsing)
├── configs/config.yaml                # All settings — single source of truth
└── .github/workflows/ci.yml           # Tests on every push
```
