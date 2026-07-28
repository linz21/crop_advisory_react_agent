"""
Hand-rolled classic ReAct agent — prompts the local LLM to reason in plain
text (Thought / Action / Action Input / Observation), parses that text to
decide which tool to call, executes it, feeds the result back as an
Observation, and repeats until the model produces a Final Answer.

WHY NOT LANGGRAPH'S PREBUILT create_react_agent: that helper calls
`model.bind_tools()`, a method that exists on LangChain chat models with
NATIVE structured tool-calling support (e.g. OpenAI/Anthropic function
calling). A raw HuggingFace text-completion pipeline — which is what a
local open-source model like Qwen2.5 is, wrapped simply — doesn't
implement that interface. This was discovered empirically (an
AttributeError at agent construction time), not assumed upfront, and led
to this alternative implementation: the classic, framework-agnostic ReAct
pattern from the original ReAct paper, which works with ANY instruction-
following model via plain text parsing rather than a native tool-calling
API.

HONEST NOTE ON MODEL CAPABILITY: whether a 1.5B-parameter local model can
reliably follow this Thought/Action/Action Input format, choose the
correct tool, and know when to stop is NOT assumed to work well — see
README's Results section for actual test outcomes once run.

Usage:
    from src.agent.react_agent import ReactAgent
    agent = ReactAgent()
    answer = agent.run("What corn yield should I expect in Illinois in 2024?")
"""

import json
import logging
import re

import yaml

from src.tools.yield_prediction_tool import predict_corn_yield
from src.tools.literature_search_tool import search_literature

log = logging.getLogger(__name__)

TOOLS = {
    "predict_corn_yield": predict_corn_yield,
    "search_literature": search_literature,
}

REACT_PROMPT_TEMPLATE = """{system_prompt}

You have access to the following tools:

predict_corn_yield: Predict corn yield (bushels per acre) for a given state \
and year, with a 95% confidence interval. Input should be a JSON object \
like {{"year": 2024, "state": "Illinois", "planted_acres": 61300}} \
(planted_acres, yield_bu_per_acre_lag1, yield_3yr_avg are optional).

search_literature: Search agronomic research literature and return a cited \
answer. Input should be a JSON object like {{"question": "How does nitrogen \
timing affect corn yield?"}}.

Use the following format exactly:

Question: the input question you must answer
Thought: think about whether you need a tool, and if so, which one
Action: the tool name, one of [predict_corn_yield, search_literature]
Action Input: a JSON object with the tool's arguments
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original question

IMPORTANT: do NOT include a numbered reference list, citation list, or a
"Sources:" section of your own in the Final Answer. The system automatically
appends the real, correct source list after your Final Answer — adding your
own creates confusing duplicate or inconsistent citations. Just write the
answer itself; sources are handled separately.

Begin!

Question: {question}
Thought:"""


class LocalQwenPipeline:
    """
    Loads the local Qwen model once (module-level singleton), used as
    the plain text completion engine for the ReAct loop — no LangChain
    chat-model interface required, since this loop parses raw text itself.
    """
    _pipe = None

    def __init__(self, model_name: str, max_new_tokens: int = 512, torch_dtype: str = "float32"):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype

    def _load(self):
        if LocalQwenPipeline._pipe is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
            dtype = dtype_map.get(self.torch_dtype, torch.float32)

            log.info(f"Loading local LLM for agent reasoning: {self.model_name} (dtype={self.torch_dtype}) ...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype)
            LocalQwenPipeline._pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens, temperature=0.01,
                do_sample=False, top_k=None, return_full_text=False,
                # IMPORTANT LESSON FROM A REAL TEST: repetition_penalty and
                # no_repeat_ngram_size penalize repeated tokens across the
                # ENTIRE input sequence, including the PROMPT itself — not
                # just newly-generated text. Our prompt necessarily
                # contains exact strings the model MUST faithfully
                # reproduce (tool names like "predict_corn_yield", example
                # values like "2024") for our regex/JSON parsing to work.
                # An earlier attempt with no_repeat_ngram_size=3 and
                # repetition_penalty=1.3 fixed the original looping
                # problem but caused a WORSE one: the model corrupted
                # "2024" into subscript unicode digits and "predict_corn_
                # yield" into "predict_corns_yield" to technically avoid
                # repeating those exact strings. no_repeat_ngram_size
                # (a hard block) was removed entirely; repetition_penalty
                # kept much softer, as a gentle nudge rather than a
                # constraint that fights against legitimate reproduction.
                repetition_penalty=1.05,
            )
        return LocalQwenPipeline._pipe

    def generate(self, prompt: str) -> str:
        pipe = self._load()
        return pipe(prompt)[0]["generated_text"]


def _truncate_hallucinated_continuation(text: str) -> str:
    """
    Small local models often don't reliably stop generating right after
    "Action Input: {...}" the way larger models trained/served with an
    explicit stop-sequence do — they instead continue generating, often
    hallucinating their OWN fake "Observation:", "Thought:", and even
    "Final Answer:" in the same completion, using invented data that has
    nothing to do with the real tool result. This was found empirically:
    a real test run showed the model fabricating an entire fake API
    response (soybean prices, income calculations) that bears no
    resemblance to what predict_corn_yield actually returns.

    This function truncates the model's raw output at the first sign of
    that hallucinated continuation — right after the Action Input line —
    so a fake self-generated Observation/Final Answer is NEVER trusted,
    regardless of how confident or well-formatted it looks. Must be
    called BEFORE checking for a final answer or parsing an action.
    """
    # If the model included its own "Observation:" after an Action Input,
    # everything from there onward is hallucinated (we haven't called the
    # real tool yet) — cut it off.
    obs_idx = text.find("\nObservation:")
    if obs_idx != -1:
        return text[:obs_idx]
    return text


def _parse_action(text: str) -> tuple[str, dict] | None:
    """
    Parses the model's output for an Action + Action Input pair. Returns
    None if no valid action is found (e.g. the model went straight to a
    Final Answer, or produced malformed output — the caller handles both
    cases).
    """
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)

    if not action_match or not input_match:
        return None

    action_name = action_match.group(1).strip()
    raw_input = input_match.group(1).strip()

    try:
        action_input = json.loads(raw_input)
    except json.JSONDecodeError:
        log.warning(f"Could not parse Action Input as JSON: {raw_input!r}")
        return None

    return action_name, action_input


def _parse_final_answer(text: str) -> str | None:
    """
    Extracts the Final Answer text, stopping at the first sign the model
    kept generating past it — a new "Question:" or "Thought:" line
    indicating hallucinated continuation into a fabricated follow-up
    exchange the user never asked for. Found empirically: a real test run
    showed the model inventing an entirely new, unprompted question
    ("What is the predicted corn yield for Illinois...") immediately after
    a legitimate Final Answer, which the original greedy regex
    (`(.*)`with DOTALL) would have silently included as part of the
    returned answer.
    """
    match = re.search(
        r"Final Answer:\s*(.*?)(?:\n\s*Question:|\n\s*Thought:|\Z)",
        text, re.DOTALL
    )
    return match.group(1).strip() if match else None


def _extract_sources(observation_text: str) -> list[str]:
    """
    Parses the "Sources: ..." line out of a search_literature tool's
    Observation text (see literature_search_tool.py — it always formats
    sources as a comma-separated list after a "Sources:" prefix).

    Used to build a code-guaranteed citation list, rather than trusting
    the model to faithfully include real citations in its own Final
    Answer text — found empirically that it sometimes doesn't (a real
    test run produced "(Citations from literature search)" as a
    placeholder instead of the actual paper titles, even though the real
    Observation did contain proper sources).
    """
    match = re.search(r"Sources:\s*(.+)", observation_text, re.DOTALL)
    if not match:
        return []
    raw = match.group(1).strip()
    # Split ONLY at a comma immediately following a closing paren with a
    # 4-digit year — e.g. "...(2025), Next title..." — NOT every comma.
    # Found empirically: paper titles themselves often contain commas
    # (e.g. "Enhance Soil Quality, Microbial Diversity, and Crop
    # Productivity..." is ONE title), and a naive split-on-every-comma
    # fragmented single titles into multiple bogus "sources".
    parts = re.split(r"(?<=\(\d{4}\))\s*,\s*", raw)
    return [s.strip() for s in parts if s.strip()]


class ReactAgent:
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.llm = LocalQwenPipeline(
            model_name=self.cfg["llm"]["model_name"],
            max_new_tokens=self.cfg["llm"]["max_new_tokens"],
            torch_dtype=self.cfg["llm"].get("torch_dtype", "float32"),
        )
        self.max_iterations = self.cfg["agent"]["max_iterations"]
        self.system_prompt = self.cfg["agent"]["system_prompt"]

    def run(self, question: str, verbose: bool = True) -> dict:
        """
        Runs the ReAct loop for a single question.

        Returns a dict with:
            answer: the final answer text, with a code-guaranteed
                    "Sources consulted" section appended if any
                    search_literature calls returned real citations
                    during this run (see _extract_sources — the model
                    isn't fully trusted to include these faithfully on
                    its own, since a real test showed it sometimes
                    drops them in favor of a vague placeholder)
            transcript: the full raw Thought/Action/Observation text, for
                        inspection — useful when the loop doesn't behave
                        as expected, since it shows the model's actual
                        reasoning rather than just a final result
            iterations: how many tool calls were made
        """
        prompt = REACT_PROMPT_TEMPLATE.format(
            system_prompt=self.system_prompt, question=question
        )
        full_transcript = prompt
        collected_sources: list[str] = []

        def _finalize(answer: str) -> str:
            """Appends a code-guaranteed source list, if any were collected,
            regardless of whether the model's own answer already included them."""
            if not collected_sources:
                return answer
            unique_sources = list(dict.fromkeys(collected_sources))  # dedupe, preserve order
            sources_block = "\n".join(f"  - {s}" for s in unique_sources)
            return f"{answer}\n\nSources consulted:\n{sources_block}"

        for i in range(self.max_iterations):
            raw_generated = self.llm.generate(full_transcript)

            # CRITICAL: truncate any hallucinated continuation (fake
            # Observation/Thought/Final Answer the model invented itself)
            # BEFORE checking for anything else — see function docstring
            # for why this was necessary (found via an actual test run
            # where the model fabricated an entire fake tool result).
            generated = _truncate_hallucinated_continuation(raw_generated)

            if verbose:
                print(f"\n--- Iteration {i+1} ---\n{generated}")
                if generated != raw_generated:
                    print(f"[Truncated {len(raw_generated) - len(generated)} chars of "
                         f"hallucinated continuation the model generated on its own]")

            final_answer = _parse_final_answer(generated)
            if final_answer:
                full_transcript += generated
                return {
                    "answer": _finalize(final_answer),
                    "transcript": full_transcript,
                    "iterations": i + 1,
                }

            action = _parse_action(generated)
            if action is None:
                # Model didn't produce a parseable action OR final answer —
                # stop here rather than looping indefinitely on malformed output
                log.warning(f"Could not parse action or final answer at iteration {i+1}")
                fallback = generated.strip() or "The agent did not produce a valid response."
                return {
                    "answer": _finalize(fallback),
                    "transcript": full_transcript + generated,
                    "iterations": i + 1,
                }

            action_name, action_input = action
            tool = TOOLS.get(action_name)

            if tool is None:
                observation = f"Error: unknown tool '{action_name}'. Available tools: {list(TOOLS.keys())}"
            else:
                try:
                    observation = tool.invoke(action_input)
                    if action_name == "search_literature":
                        collected_sources.extend(_extract_sources(observation))
                except Exception as e:
                    observation = f"Error calling {action_name}: {e}"

            full_transcript += generated + f"\nObservation: {observation}\nThought:"

        return {
            "answer": _finalize("The agent reached the maximum number of reasoning steps without a final answer."),
            "transcript": full_transcript,
            "iterations": self.max_iterations,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration output")
    args = parser.parse_args()

    agent = ReactAgent()
    result = agent.run(args.question, verbose=not args.quiet)

    print(f"\n{'='*70}")
    print(f"Q: {args.question}")
    print(f"{'='*70}")
    print(f"\nA: {result['answer']}")
    print(f"\n(Completed in {result['iterations']} iteration(s))")


if __name__ == "__main__":
    main()
