"""
Literature search tool — wraps Project 3's retrieval + generation pipeline
as a LangChain/LangGraph-compatible tool for the ReAct agent.

Calls Project 3's code IN-PROCESS (direct Python import), not over HTTP —
unlike the yield prediction tool, which calls Project 1's already-deployed
API. This is a deliberate difference: Project 3 was never deployed as a
standing API (only as a Hugging Face Space with a Gradio UI, not a
callable REST endpoint), so importing its retriever/generator modules
directly is simpler than standing up a new API just for this agent to
call. See Known Gaps in this project's README for the coupling this
creates between the two projects' codebases.

IMPORTANT — NAMESPACE COLLISION: both this project and Project 3 use the
top-level package name `src`. A naive `sys.path.insert()` + `from src.x
import y` does NOT work here — once Python has imported THIS project's
`src` package (which happens as soon as this file itself is imported via
`src.tools.literature_search_tool`), that `src` name is cached in
sys.modules for the rest of the process. Adding Project 3's directory to
sys.path afterward doesn't change what `src` already resolves to. The fix
below loads Project 3's specific module FILES directly via importlib,
using distinct internal names, rather than importing through the
colliding `src.` package path.

Usage:
    from src.tools.literature_search_tool import search_literature
    result = search_literature.invoke({
        "question": "How does nitrogen timing affect corn yield?"
    })
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import yaml
from langchain.tools import tool

log = logging.getLogger(__name__)

_retriever = None
_generate_answer_fn = None


def _load_module_from_path(module_name: str, file_path: Path):
    """
    Load a Python module directly from a file path, registered under a
    distinct name — avoids collisions with this project's own `src`
    package namespace. See module docstring above for why this is
    necessary rather than a simpler sys.path + package import.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # register under the unique name
    spec.loader.exec_module(module)
    return module


def _get_project3_modules():
    """
    Lazily load Project 3's retriever and generator modules by explicit
    file path (see namespace collision note above). Requires Project 3's
    repo cloned as a sibling directory (see configs/config.yaml's
    project3_path) with its own dependencies already installed.

    ALSO HANDLES A SECOND CROSS-PROJECT ISSUE: Project 3's own config.yaml
    uses paths RELATIVE to its own directory (e.g. "data/processed/
    chunks.json", "data/chroma_db") — these only resolve correctly when
    Project 3's directory is the current working directory. Since this
    tool runs from crop_advisory_react_agent's directory, we temporarily
    os.chdir() into Project 3's directory during initialization, then
    restore the original working directory afterward — same
    "swap, use, restore" pattern used for the `src` namespace collision
    above, applied to a second, independent cross-project coupling issue.
    """
    global _retriever, _generate_answer_fn
    if _retriever is None:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)

        project3_path = Path(cfg["tools"]["literature_search"]["project3_path"]).resolve()
        if not project3_path.exists():
            raise FileNotFoundError(
                f"Project 3 not found at {project3_path}. This tool requires "
                "agri_rag_literature_ga cloned as a sibling directory — see "
                "README Setup section."
            )

        this_project_src = sys.modules.get("src")
        project3_src_init = project3_path / "src" / "__init__.py"

        project3_src_spec = importlib.util.spec_from_file_location(
            "src", project3_src_init, submodule_search_locations=[str(project3_path / "src")]
        )
        project3_src_module = importlib.util.module_from_spec(project3_src_spec)
        sys.modules["src"] = project3_src_module  # temporarily override

        original_cwd = os.getcwd()

        try:
            os.chdir(project3_path)  # so project3's relative data paths resolve correctly

            retriever_module = _load_module_from_path(
                "project3_retriever", project3_path / "src" / "retrieval" / "retriever.py"
            )
            generator_module = _load_module_from_path(
                "project3_generator", project3_path / "src" / "generation" / "generator.py"
            )

            with open(project3_path / "configs" / "config.yaml") as f:
                project3_cfg = yaml.safe_load(f)

            _retriever = retriever_module.HybridRetriever(project3_cfg)
            # Force the embedder to load NOW, while project3's `src` is still
            # active in sys.modules AND we're still in project3's directory
            # — both conditions matter here (see docstring above).
            _retriever._get_embedder()
            _generate_answer_fn = generator_module.generate_answer
            log.info(f"Project 3 retriever and generator initialized from {project3_path}")

        finally:
            os.chdir(original_cwd)
            # Restore THIS project's own `src` so nothing else breaks
            if this_project_src is not None:
                sys.modules["src"] = this_project_src
            else:
                sys.modules.pop("src", None)

    return _retriever, _generate_answer_fn


@tool
def search_literature(question: str) -> str:
    """
    Search agronomic research literature (478 real PubMed papers on corn
    yield, precision agriculture, and crop science) and return a grounded,
    cited answer. Use this tool when the user asks about research findings,
    agronomic mechanisms, best practices, or "why"/"how" questions that
    aren't simple numeric yield forecasts.

    Args:
        question: The research question to search for

    Returns:
        A cited answer synthesized from relevant research paper abstracts.
    """
    try:
        retriever, generate_answer_fn = _get_project3_modules()
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        log.error(f"Failed to initialize literature search: {e}")
        return f"Literature search is currently unavailable: {e}"

    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        top_k = cfg["tools"]["literature_search"]["top_k"]

        chunks = retriever.search(question, top_k=top_k)
        if not chunks:
            return "No relevant research papers were found for this question."

        project3_path = Path(cfg["tools"]["literature_search"]["project3_path"]).resolve()
        with open(project3_path / "configs" / "config.yaml") as f:
            project3_cfg = yaml.safe_load(f)

        result = generate_answer_fn(question, chunks, project3_cfg)

        sources = ", ".join(f"{s['title']} ({s['year']})" for s in result["sources"])
        return f"{result['answer']}\n\nSources: {sources}"

    except Exception as e:
        log.error(f"Literature search tool error: {e}")
        return f"Error searching literature: {e}"
