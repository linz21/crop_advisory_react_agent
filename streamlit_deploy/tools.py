"""
Yield prediction tool for the Streamlit Cloud deployment — identical logic
to src/tools/yield_prediction_tool.py, duplicated here rather than
imported, to keep this deployment package fully self-contained (same
reasoning as Project 2's hf_space/ folder: a self-contained deployment
package proved much easier to debug than trying to cleverly share code
across the local-dev and deployed versions).

NOTE: unlike the main project's version, this is a PLAIN function, not
wrapped in langchain's @tool decorator — the decorator pulls in the
entire langchain dependency tree for a deployment that doesn't otherwise
need it, and risks a repeat of the real guardrails-ai/langchain-core
version conflict hit in the main project. This deployment's agent.py
calls tools directly as `tool(**kwargs)` rather than `tool.invoke(dict)`.
"""

import logging

import requests

log = logging.getLogger(__name__)

YIELD_API_URL = "http://54.214.151.133:8000/predict"


def predict_corn_yield(year: int, state: str, planted_acres: float = 50000,
                       yield_bu_per_acre_lag1: float = None,
                       yield_3yr_avg: float = None) -> str:
    """
    Predict corn yield (bushels per acre) for a given state and year, with a
    95% confidence interval.
    """
    payload = {"year": year, "state": state, "planted_acres": planted_acres}
    if yield_bu_per_acre_lag1 is not None:
        payload["yield_bu_per_acre_lag1"] = yield_bu_per_acre_lag1
    if yield_3yr_avg is not None:
        payload["yield_3yr_avg"] = yield_3yr_avg

    try:
        response = requests.post(YIELD_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()

        return (
            f"Predicted corn yield for {state} in {year}: "
            f"{data['predicted_yield_bu_per_acre']} bu/acre "
            f"(95% CI: {data['ci_lower']}-{data['ci_upper']} bu/acre)."
        )

    except requests.exceptions.ConnectionError:
        log.warning(f"Yield prediction API unreachable at {YIELD_API_URL}")
        return (
            "The yield prediction service is currently unavailable "
            "(the demo API may be paused to manage cloud costs). "
            "Unable to provide a numeric yield forecast right now."
        )
    except requests.exceptions.Timeout:
        return "The yield prediction service timed out. Please try again."
    except Exception as e:
        log.error(f"Yield prediction tool error: {e}")
        return f"Error getting yield prediction: {e}"
