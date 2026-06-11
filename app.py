"""Streamlit UI for the telecom tower lease vetting agent."""

from __future__ import annotations

from typing import Any

import streamlit as st
from google.genai.errors import ClientError, ServerError

from agent import LeaseVettingAgent
from config import get_api_key, get_api_key_fingerprint, load_env

load_env()

EXAMPLE_REQUESTS: dict[str, str] = {
    "Approved (TWR-101)": (
        "Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters "
        "on Tower TWR-101."
    ),
    "Rejected (over capacity)": (
        "Operator Etisalat wants to mount a 100kg microwave dish at a height of "
        "35 meters on Tower TWR-101."
    ),
    "Rejected (unknown tower)": (
        "Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters "
        "on Tower TWR-999."
    ),
}

PLACEHOLDER = (
    "Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters "
    "on Tower TWR-101."
)


@st.cache_resource
def get_agent(api_key: str) -> LeaseVettingAgent:
    """Create and cache the vetting agent for the session (keyed on API key)."""
    return LeaseVettingAgent(api_key=api_key)


def render_verdict(verdict: dict[str, Any]) -> None:
    """Render a human-readable verdict card."""
    status = verdict.get("status", "UNKNOWN")

    if status == "APPROVED":
        st.success(f"**{status}**")
    else:
        st.error(f"**{status}**")

    st.info(verdict.get("reason", "No reason provided."))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Operator", verdict.get("operator") or "—")
    with col2:
        st.metric("Tower ID", verdict.get("tower_id") or "—")

    checks = verdict.get("checks_run", [])
    if checks:
        st.subheader("Checks performed")
        rows = []
        for check in checks:
            passed = check.get("passed", False)
            rows.append({
                "Check": check.get("name", "unknown"),
                "Result": "Passed" if passed else "Failed",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


def show_auth_error(exc: ClientError, key_fingerprint: str) -> None:
    """Display actionable guidance for authentication failures."""
    st.error("Gemini API authentication failed (401 UNAUTHENTICATED).")
    st.warning(f"Key used by this app: **{key_fingerprint}**")
    st.markdown(
        "Your API key was rejected by Google. Try these steps:\n"
        "1. Create a **new API key** at [Google AI Studio](https://aistudio.google.com/apikey)\n"
        "2. Update `.env` in the project folder:\n"
        "   `GEMINI_API_KEY=your-new-key`\n"
        "3. Click **Clear cache & reload** in the sidebar (or restart Streamlit)\n"
        "4. Check Windows **Environment Variables** — remove any old `GOOGLE_API_KEY` "
        "that may override your `.env` file\n"
        "5. If only `AQ.` keys fail, create a key in a **new AI Studio project**"
    )
    with st.expander("Technical details"):
        st.code(str(exc))


def main() -> None:
    st.set_page_config(
        page_title="Tower Lease Vetting",
        layout="centered",
    )

    st.title("Tower Lease Vetting")
    st.caption(
        "Submit a plain-text lease request to check tower capacity and regional policy."
    )

    api_key: str | None = None
    try:
        api_key = get_api_key()
    except ValueError:
        pass

    with st.sidebar:
        st.header("Example requests")
        for label, text in EXAMPLE_REQUESTS.items():
            if st.button(label, use_container_width=True):
                st.session_state["request_text"] = text

        st.divider()
        st.markdown("**Setup**")
        if api_key:
            st.success(f"API key: {get_api_key_fingerprint()}")
            if st.button("Clear cache & reload", use_container_width=True):
                get_agent.clear()
                st.rerun()
        else:
            st.warning("Set `GEMINI_API_KEY` in the project `.env` file.")

    if "request_text" not in st.session_state:
        st.session_state["request_text"] = ""

    request_text = st.text_area(
        "Lease request",
        value=st.session_state["request_text"],
        placeholder=PLACEHOLDER,
        height=80,
        label_visibility="collapsed",
    )
    st.session_state["request_text"] = request_text

    submitted = st.button("Vet Request", type="primary")

    if not submitted:
        return

    if not request_text.strip():
        st.warning("Please enter a lease request.")
        return

    try:
        api_key = get_api_key()
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        agent = get_agent(api_key)
        with st.spinner("Analyzing lease request..."):
            verdict = agent.vet_lease(request_text.strip())
    except ValueError as exc:
        st.error(str(exc))
        return
    except ClientError as exc:
        error_text = str(exc)
        if "401" in error_text or "UNAUTHENTICATED" in error_text:
            get_agent.clear()
            show_auth_error(exc, get_api_key_fingerprint())
        else:
            st.error(f"Gemini API error: {exc}")
            st.info(
                "Tip: if you hit quota limits, use `GEMINI_MODEL=gemini-flash-latest` in `.env`."
            )
        return
    except ServerError as exc:
        st.error(f"Gemini API temporarily unavailable: {exc}")
        st.info("Please retry in a few seconds.")
        return

    st.divider()
    render_verdict(verdict)


if __name__ == "__main__":
    main()
