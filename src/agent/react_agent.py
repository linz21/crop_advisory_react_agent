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
import uuid

import yaml
from langsmith import traceable

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
{memory_context}
Begin!
{memory_check_reminder}
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

    @traceable(run_type="llm", name="local_qwen_generate")
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
        r"Final Answer:\s*(.*?)(?:\n\s*Question:|\n\s*Thought:|\n\s*Final Answer:|\Z)",
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


class AnthropicLLM:
    """
    Drop-in alternative to LocalQwenPipeline, using the Anthropic API
    instead of a local model — same .generate(prompt) interface, so the
    rest of the ReAct loop code doesn't need to change at all.

    ADDED SPECIFICALLY to test whether a frontier model fixes a real,
    reproducible limitation found in BOTH Qwen2.5-1.5B and Qwen3-4B: given
    real prior conversation context showing a previous answer contained
    ONLY a yield number, both models still fabricated a false claim that
    the previous answer had mentioned "practices" or "nitrogen timing" —
    a genuine memory-verification failure that survived two different,
    reasonable prompt-engineering attempts (see _build_memory_check_
    reminder's docstring for both failed attempts). This is a real,
    empirical test of whether that failure is a MODEL CAPABILITY limit
    (which a frontier model should fix) or something else entirely
    (which it wouldn't).

    REQUIRES a genuine Anthropic API key from console.anthropic.com with
    billing enabled — NOT the same as a Claude Pro/Max subscription used
    for claude.ai or Claude Code, which does not provide API access. This
    is a separate product with separate billing (same distinction
    documented in Project 3, which hit this exact confusion originally).

    Usage:
        export ANTHROPIC_API_KEY="sk-ant-..."
        # then set llm.provider: "anthropic" in configs/config.yaml
    """

    def __init__(self, model_name: str = "claude-sonnet-4-5", max_tokens: int = 512):
        import os
        import anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY environment variable not set. This requires "
                "a genuine API key from console.anthropic.com with billing "
                "enabled — separate from a Claude Pro/Max subscription. "
                "See README Setup section."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens

    @traceable(run_type="llm", name="anthropic_generate")
    def generate(self, prompt: str) -> str:
        """
        Same signature as LocalQwenPipeline.generate() — takes the full
        accumulated ReAct transcript as a single string prompt, returns
        the model's continuation as a string. Uses the Messages API's
        single-user-turn pattern (the whole ReAct transcript is the
        "user" content) since our hand-rolled loop manages the actual
        conversation structure itself via plain text, not the API's
        multi-turn message format.
        """
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""


class ReactAgent:
    def __init__(self, config_path: str = "configs/config.yaml", session_id: str = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        provider = self.cfg["llm"].get("provider", "local")
        if provider == "anthropic":
            self.llm = AnthropicLLM(
                model_name=self.cfg["llm"].get("anthropic_model", "claude-sonnet-4-5"),
                max_tokens=self.cfg["llm"]["max_new_tokens"],
            )
        else:
            self.llm = LocalQwenPipeline(
                model_name=self.cfg["llm"]["model_name"],
                max_new_tokens=self.cfg["llm"]["max_new_tokens"],
                torch_dtype=self.cfg["llm"].get("torch_dtype", "float32"),
            )
        self.max_iterations = self.cfg["agent"]["max_iterations"]
        self.system_prompt = self.cfg["agent"]["system_prompt"]

        # session_id identifies THIS conversation for short-term (Redis)
        # memory. If not provided, a random one is generated — meaning
        # short-term history won't carry over between separate ReactAgent
        # instances unless the same session_id is explicitly reused.
        self.session_id = session_id or str(uuid.uuid4())

        # Both memory types are OPTIONAL and degrade gracefully — the
        # agent still works with neither configured (e.g. no REDIS_HOST
        # set), same pattern as the rest of this project's tools.
        self._short_term_memory = None
        self._long_term_memory = None

    def _get_short_term_memory(self):
        if self._short_term_memory is None:
            try:
                from src.memory.redis_memory import SessionMemory
                ttl = self.cfg["memory"].get("session_ttl_seconds", 3600)
                self._short_term_memory = SessionMemory(self.session_id, ttl_seconds=ttl)
            except Exception as e:
                log.warning(f"Short-term (Redis) memory unavailable: {e}")
                self._short_term_memory = False  # sentinel: tried and failed, don't retry
        return self._short_term_memory or None

    def _get_long_term_memory(self):
        if self._long_term_memory is None:
            try:
                from src.memory.long_term_memory import LongTermMemory
                self._long_term_memory = LongTermMemory()
            except Exception as e:
                log.warning(f"Long-term (vector store) memory unavailable: {e}")
                self._long_term_memory = False
        return self._long_term_memory or None

    def _build_memory_context(self, question: str) -> str:
        """
        Builds the optional memory-context block inserted into the prompt:
        recent turns from THIS session (short-term, Redis) plus
        semantically relevant past exchanges from ANY session (long-term,
        vector store). Returns an empty string if neither is available or
        neither has anything relevant — the prompt template handles an
        empty memory_context cleanly (just an extra blank line).
        """
        sections = []

        short_term = self._get_short_term_memory()
        if short_term is not None:
            history = short_term.get_history()
            if history:
                recent = history[-4:]  # last 2 exchanges (user+assistant pairs)
                lines = [f"{h['role']}: {h['content']}" for h in recent]
                sections.append(
                    "Recent conversation in this session:\n" + "\n".join(lines)
                )

        long_term = self._get_long_term_memory()
        if long_term is not None:
            relevant = long_term.search_relevant_history(question, top_k=2)
            if relevant:
                lines = [f"- Q: {r['question']}\n  A: {r['answer'][:200]}" for r in relevant]
                sections.append(
                    "Potentially relevant past interactions (may be from a "
                    "different session):\n" + "\n".join(lines)
                )

        if not sections:
            return ""

        return "\n" + "\n\n".join(sections) + "\n"

    def _build_memory_check_reminder(self, has_memory_context: bool) -> str:
        """
        A second, separate prompt-engineering attempt at the same problem
        _build_memory_context's instruction was meant to solve — kept as
        its own function/placeholder specifically to place it IMMEDIATELY
        adjacent to "Question: {question}", not several paragraphs earlier
        (a first attempt embedded the instruction inside memory_context,
        separated from the question by "Begin!" and a blank line — that
        version was tested and FAILED: the model still fabricated a false
        claim about what a prior turn contained, despite the instruction
        being present in the prompt).

        This version: (1) moves the instruction to be the literal last
        text before the question, testing whether recency effects in a
        long prompt were the issue, and (2) replaces the abstract "check
        first" wording with a concrete worked example, since small models
        often follow concrete examples more reliably than abstract rules.
        """
        if not has_memory_context:
            return ""

        return (
            "REMINDER: Before answering, look at the conversation history "
            "above. Example: if it shows a previous question only received "
            "a yield number with no mention of farming practices, and this "
            "new question asks about \"the practices mentioned,\" the "
            "correct response is: \"The previous answer only provided a "
            "yield number and did not mention specific practices.\" Do not "
            "invent or assume content that is not actually shown above.\n\n"
        )
        return "\n" + "\n\n".join(sections) + "\n"

    @traceable(run_type="chain", name="crop_advisory_agent_run")
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
        memory_context = self._build_memory_context(question)
        memory_check_reminder = self._build_memory_check_reminder(has_memory_context=bool(memory_context))
        prompt = REACT_PROMPT_TEMPLATE.format(
            system_prompt=self.system_prompt, question=question,
            memory_context=memory_context, memory_check_reminder=memory_check_reminder,
        )
        full_transcript = prompt
        collected_sources: list[str] = []

        def _finalize(answer: str) -> str:
            """Appends a code-guaranteed source list, if any were collected,
            regardless of whether the model's own answer already included
            them. Also saves this Q&A exchange to both memory stores
            (best-effort — a memory save failure shouldn't break the
            actual answer being returned to the user)."""
            if collected_sources:
                unique_sources = list(dict.fromkeys(collected_sources))  # dedupe, preserve order
                sources_block = "\n".join(f"  - {s}" for s in unique_sources)
                final_text = f"{answer}\n\nSources consulted:\n{sources_block}"
            else:
                final_text = answer

            try:
                short_term = self._get_short_term_memory()
                if short_term is not None:
                    short_term.add_message("user", question)
                    short_term.add_message("assistant", final_text)
            except Exception as e:
                log.warning(f"Failed to save to short-term memory: {e}")

            try:
                long_term = self._get_long_term_memory()
                if long_term is not None:
                    long_term.add_interaction(question, final_text, self.session_id)
            except Exception as e:
                log.warning(f"Failed to save to long-term memory: {e}")

            # Guardrails validation — detection + audit logging, NOT
            # automatic blocking (see build_guard()'s docstring for why:
            # a false positive silently altering a farmer's answer is its
            # own risk). Best-effort, same reasoning as memory saves above.
            #
            # IMPORTANT: validate the RAW model `answer`, not `final_text`
            # — final_text includes our OWN code-appended "Sources
            # consulted:" block, which would otherwise always trip the
            # self-citation validator's "^Sources?:" pattern on every
            # single successful answer (a real bug caught while writing
            # tests, fixed here rather than left in).
            guardrails_passed, guardrails_issues = True, []
            try:
                from src.guardrails.validators import validate_response
                guardrails_passed, guardrails_issues = validate_response(answer)
                if not guardrails_passed:
                    log.warning(f"Guardrails flagged this response: {guardrails_issues}")
            except Exception as e:
                log.warning(f"Guardrails validation failed to run: {e}")

            try:
                from src.guardrails.audit_log import log_interaction
                log_interaction(
                    session_id=self.session_id, question=question, answer=final_text,
                    iterations=i + 1, guardrails_passed=guardrails_passed,
                    guardrails_issues=guardrails_issues,
                    llm_provider=self.cfg["llm"].get("provider", "local"),
                )
            except Exception as e:
                log.warning(f"Failed to write audit log: {e}")

            return final_text

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
    parser.add_argument("--session-id", default=None,
                        help="Reuse the same session ID across multiple calls to test "
                             "short-term memory persistence within one conversation")
    args = parser.parse_args()

    agent = ReactAgent(session_id=args.session_id)
    result = agent.run(args.question, verbose=not args.quiet)

    print(f"\n{'='*70}")
    print(f"Q: {args.question}")
    print(f"Session ID: {agent.session_id}")
    print(f"{'='*70}")
    print(f"\nA: {result['answer']}")
    print(f"\n(Completed in {result['iterations']} iteration(s))")


if __name__ == "__main__":
    main()
