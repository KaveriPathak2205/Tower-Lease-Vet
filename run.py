"""CLI entry point for the telecom tower lease vetting agent."""

from __future__ import annotations

import json
import sys

from google.genai.errors import ClientError, ServerError

from agent import LeaseVettingAgent
from config import load_env

load_env()


def read_request(argv: list[str]) -> str:
    """
    Read the lease request from CLI arguments or stdin.

    Args:
        argv: Command-line arguments (excluding script name).

    Returns:
        Plain-text lease request.

    Raises:
        SystemExit: If no input is provided.
    """
    if argv:
        return " ".join(argv).strip()
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    print(
        "Usage: python run.py \"<lease request text>\"",
        file=sys.stderr,
    )
    print(
        "   or: echo \"<lease request text>\" | python run.py",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> None:
    """Run the vetting agent and print the JSON verdict."""
    request = read_request(sys.argv[1:])

    try:
        agent = LeaseVettingAgent()
        result = agent.vet_lease(request)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    except ClientError as exc:
        error_text = str(exc)
        print(
            f"Gemini API error ({getattr(exc, 'status_code', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        if "401" in error_text or "UNAUTHENTICATED" in error_text:
            print(
                "Authentication failed. Create a new key at "
                "https://aistudio.google.com/apikey and update GEMINI_API_KEY in .env",
                file=sys.stderr,
            )
        elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            print(
                "Tip: gemini-2.0-flash has no free-tier quota. "
                "Use gemini-flash-latest (default) or set GEMINI_MODEL.",
                file=sys.stderr,
            )
        raise SystemExit(1) from exc
    except ServerError as exc:
        print(
            f"Gemini API temporarily unavailable: {exc}",
            file=sys.stderr,
        )
        print("Please retry in a few seconds.", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
