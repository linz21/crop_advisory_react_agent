# Deploying to Streamlit Community Cloud

This folder (`streamlit_deploy/`) is a self-contained deployment package —
everything needed lives here, with no dependency on this repo's other
folders or on Project 2 being cloned locally (it calls Project 2's
already-deployed Hugging Face Space instead — see `literature_tool_remote.py`).

## Step 0 — Confirm Project 2's API endpoint name (REQUIRED FIRST)

1. Go to https://huggingface.co/spaces/lzhang2026/agri-rag-assistant
2. Scroll to the footer and click **"Use via API"**
3. Note the exact `api_name` shown (likely `/ask_question`, but confirm)
4. If it differs from `/ask_question`, edit `literature_tool_remote.py`
   and update the `API_NAME` constant at the top of the file

## Step 1 — Push this folder to GitHub

This can live inside your existing `crop_advisory_react_agent` repo (as
this `streamlit_deploy/` subfolder) — Streamlit Cloud lets you specify a
file path within a repo, not just the repo root.

```bash
cd ~/Desktop/mle_projects/crop_advisory_react_agent
git add streamlit_deploy/
git commit -m "Add Streamlit Cloud deployment package"
git push origin main
```

## Step 2 — Create the app on Streamlit Community Cloud

1. Go to https://share.streamlit.io (or streamlit.io/cloud)
2. Sign in with GitHub
3. Click **"New app"**
4. **Repository:** `linz21/crop_advisory_react_agent`
5. **Branch:** `main`
6. **Main file path:** `streamlit_deploy/app.py`
7. Click **"Advanced settings"** before deploying (see Step 3)

## Step 3 — Set secrets (in Advanced settings, before first deploy)

Paste this into the Secrets field, filling in your real values:

```toml
ANTHROPIC_API_KEY = "sk-ant-your-real-key-here"
HF_TOKEN = "hf_your-real-token-here"
```

**`ANTHROPIC_API_KEY`:** genuine Anthropic API key with billing enabled
from console.anthropic.com — NOT a Claude Pro/Max subscription (separate
product/billing — see main README).

**`HF_TOKEN`:** a free Hugging Face token from
huggingface.co/settings/tokens (read-access is enough). REQUIRED, not
optional — a real test without this hit "You have exceeded your ZeroGPU
quota" after only a couple of literature-search calls, since anonymous
`gradio_client` connections get a much lower quota tier on Project 2's
Space. Authenticating gets a meaningfully higher quota (see
`literature_tool_remote.py`'s docstring for the exact error hit).

Top-level TOML entries (no `[section]` header) are automatically exposed
as real environment variables, which is what `os.getenv()` in this code
expects — confirmed via Streamlit's own docs, not assumed.

## Step 4 — Deploy

Click **"Deploy!"**. First build takes a few minutes (installing
dependencies). Once live, you'll get a URL like:
```
https://your-app-name.streamlit.app
```

## Known limitations of this deployment

- **No local model option** — Anthropic API only, by design (see
  `agent.py`'s docstring for why: not feasible on free-tier resources,
  and real testing found local models less reliable anyway).
- **No Redis memory, no LangSmith tracing, no Guardrails validation** —
  intentionally left out of this minimal deployment to keep the
  dependency footprint small and the deployment simple to debug. The
  full-featured version (all of these included) runs locally — see the
  main README.
- **Depends on Project 1's and Project 2's own live services staying
  up** — if either is paused/down, this deployment's corresponding tool
  will report that limitation gracefully rather than crash, but won't
  have a fallback.
- **`literature_tool_remote.py`'s response-parsing logic is defensive
  but unverified** — Project 2's Gradio function returns a specific
  shape that wasn't confirmed against the real API before writing this;
  test directly after deployment and adjust if the parsing doesn't
  extract the answer text correctly (see that file's docstring).
