"""
Yield prediction tool — wraps Project 1's corn yield prediction API as a
LangChain/LangGraph-compatible tool for the ReAct agent.

Calls the LIVE deployed API (AWS EC2) over HTTP by default. This is a real
network dependency — if Project 1's EC2 instance is stopped (e.g. to save
cost, as documented in that project's README), this tool will fail. See
Known Gaps in this project's README for how that's handled.

Usage:
    from src.tools.yield_prediction_tool import predict_corn_yield
    result = predict_corn_yield.invoke({
        "year": 2024, "state": "Illinois", "planted_acres": 61300
    })
"""

import logging

import requests
from langchain.tools import tool

log = logging.getLogger(__name__)


@tool
def predict_corn_yield(year: int, state: str, planted_acres: float = 50000,
                       yield_bu_per_acre_lag1: float = None,
                       yield_3yr_avg: float = None) -> str:
    """
    Predict corn yield (bushels per acre) for a given state and year, with a
    95% confidence interval. Use this tool when the user asks about expected
    corn yield, yield forecasts, or planning decisions based on projected
    yield.

    Args:
        year: The year to predict yield for (e.g. 2024)
        state: US state name (e.g. "Illinois")
        planted_acres: Acres planted, if known (defaults to a typical value)
        yield_bu_per_acre_lag1: Previous year's yield for this state, if known
        yield_3yr_avg: 3-year rolling average yield for this state, if known

    Returns:
        A string summarizing the predicted yield and confidence interval.
    """
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    api_url = cfg["tools"]["yield_prediction"]["api_url"]
    timeout = cfg["tools"]["yield_prediction"]["timeout_seconds"]

    payload = {"year": year, "state": state, "planted_acres": planted_acres}
    if yield_bu_per_acre_lag1 is not None:
        payload["yield_bu_per_acre_lag1"] = yield_bu_per_acre_lag1
    if yield_3yr_avg is not None:
        payload["yield_3yr_avg"] = yield_3yr_avg

    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()

        return (
            f"Predicted corn yield for {state} in {year}: "
            f"{data['predicted_yield_bu_per_acre']} bu/acre "
            f"(95% CI: {data['ci_lower']}-{data['ci_upper']} bu/acre)."
        )

    except requests.exceptions.ConnectionError:
        log.warning(f"Yield prediction API unreachable at {api_url}")
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
