"""
ReAct agent for the Streamlit Cloud deployment — Anthropic-only (no local
model support). Local models (Qwen3-4B etc.) aren't feasible on Streamlit
Cloud's free tier (limited RAM, no GPU) — and separately, real testing in
this project found local models had genuine reliability issues (fabrication
on tool failure, fabrication on memory verification) that Claude Sonnet 4.5
resolved — so Anthropic is the only, and the right, choice for this public
deployment. See the main project's README Results section for that
evidence.

Same hand-rolled ReAct loop logic as src/agent/react_agent.py (see that
file's docstrings for why LangGraph's prebuilt agent isn't used, and for
the several hallucinated-continuation edge cases this parsing handles) —
duplicated here rather than imported to keep this deployment package
self-contained.
"""

import json
import logging
import os
import re
import uuid

import anthropic

from tools import predict_corn_yield
from literature_tool_remote import search_literature

log = logging.getLogger(__name__)

TOOLS = {
    "predict_corn_yield": predict_corn_yield,
    "search_literature": search_literature,
}

SYSTEM_PROMPT = (
    "You are a crop advisory assistant for corn farmers. You have access to "
    "two tools: a yield prediction model and a research literature search "
    "tool. Use these tools to answer questions about corn yield, farming "
    "practices, and agronomic research. Always cite which tool(s) informed "
    "your answer. If a question is outside corn/agriculture, say so rather "
    "than guessing."
)

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
appends the real, correct source list after your Final Answer.

Begin!

Question: {question}
Thought:"""


def _truncate_hallucinated_continuation(text: str) -> str:
    obs_idx = text.find("\nObservation:")
    return text[:obs_idx] if obs_idx != -1 else text


def _parse_action(text: str) -> tuple[str, dict] | None:
    action_match = re.search(r"Action:\s*(\w+)", text)
    input_match = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not action_match or not input_match:
        return None
    try:
        return action_match.group(1).strip(), json.loads(input_match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _parse_final_answer(text: str) -> str | None:
    match = re.search(
        r"Final Answer:\s*(.*?)(?:\n\s*Question:|\n\s*Thought:|\n\s*Final Answer:|\Z)",
        text, re.DOTALL
    )
    return match.group(1).strip() if match else None


def _extract_sources(observation_text: str) -> list[str]:
    """
    Tries the numbered-list format FIRST — confirmed via real testing that
    this is what Project 2's REMOTE/deployed API actually returns
    ("**Sources:**" header + "1. Title (Year)" lines), which is DIFFERENT
    from the comma-separated format the LOCAL in-process version uses
    ("Sources: Title (Year), Title (Year)"). A real test with the remote
    tool showed the original comma-only version swallowed the entire
    numbered block as one malformed entry, since there's no comma-after-
    year to split on in a newline-separated numbered list. Falls back to
    the comma-separated pattern for robustness, though the remote API is
    the only one this deployment actually uses.
    """
    numbered = re.findall(r"^\s*\d+\.\s+(.+?\(\d{4}\))\s*$", observation_text, re.MULTILINE)
    if numbered:
        return [s.strip() for s in numbered]

    match = re.search(r"Sources:\s*(.+)", observation_text, re.DOTALL)
    if not match:
        return []
    raw = match.group(1).strip()
    parts = re.split(r"(?<=\(\d{4}\))\s*,\s*", raw)
    return [s.strip() for s in parts if s.strip()]


class DeployedAgent:
    def __init__(self, session_id: str = None, max_iterations: int = 5):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = "claude-sonnet-4-5"
        self.max_iterations = max_iterations
        self.session_id = session_id or str(uuid.uuid4())

    def _generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name, max_tokens=512, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    def run(self, question: str) -> dict:
        prompt = REACT_PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)
        full_transcript = prompt
        collected_sources: list[str] = []

        def _finalize(answer: str) -> str:
            if not collected_sources:
                return answer
            unique = list(dict.fromkeys(collected_sources))
            block = "\n".join(f"  - {s}" for s in unique)
            return f"{answer}\n\nSources consulted:\n{block}"

        for i in range(self.max_iterations):
            raw = self._generate(full_transcript)
            generated = _truncate_hallucinated_continuation(raw)

            final_answer = _parse_final_answer(generated)
            if final_answer:
                return {"answer": _finalize(final_answer), "iterations": i + 1}

            action = _parse_action(generated)
            if action is None:
                fallback = generated.strip() or "The agent did not produce a valid response."
                return {"answer": _finalize(fallback), "iterations": i + 1}

            action_name, action_input = action
            tool = TOOLS.get(action_name)
            if tool is None:
                observation = f"Error: unknown tool '{action_name}'."
            else:
                try:
                    observation = tool(**action_input)
                    if action_name == "search_literature":
                        collected_sources.extend(_extract_sources(observation))
                except Exception as e:
                    observation = f"Error calling {action_name}: {e}"

            full_transcript += generated + f"\nObservation: {observation}\nThought:"

        return {
            "answer": _finalize("The agent reached the maximum number of reasoning steps."),
            "iterations": self.max_iterations,
        }
