"""Google Gemini-powered agent for vetting telecom tower lease requests."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google import genai
from google.genai import types

from tools import NO_POLICY_FOUND, TOWER_NOT_FOUND, execute_tool
from config import get_api_key

DEFAULT_MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = """You are a telecom tower lease vetting agent for a municipality.

When given a plain-text lease request, you must:
1. Extract: operator_name, tower_id, equipment_weight_kg, mounting_height_m.
2. Call BOTH tools before making a final decision:
   - check_tower_capacity(tower_id, weight_kg)
   - check_regional_policy(tower_id, height_m, weight_kg)
3. After all tool results are available, respond with ONLY a valid JSON object (no markdown, no extra text) with these fields:
   - status: "APPROVED" or "REJECTED"
   - reason: one clear sentence explaining the decision
   - operator: the operator name from the request
   - tower_id: the tower identifier
   - checks_run: list of objects with "name" (check name) and "passed" (boolean)

Decision rules:
- APPROVED only if ALL checks passed.
- REJECTED if any check failed.
- If a tool returns error "Tower not found in inventory.", status must be REJECTED with reason exactly "Tower not found in inventory."
- If a tool returns error "No policy found for region.", status must be REJECTED with reason exactly "No policy found for region."
"""

TOOL_DEFINITIONS = [
    types.FunctionDeclaration(
        name="check_tower_capacity",
        description=(
            "Check if a tower has sufficient remaining weight capacity for new equipment. "
            "Looks up the tower in the inventory and verifies that "
            "current_weight + new_weight <= max_allowed_weight."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tower_id": types.Schema(
                    type=types.Type.STRING,
                    description="Tower identifier, e.g. TWR-101",
                ),
                "weight_kg": types.Schema(
                    type=types.Type.NUMBER,
                    description="Weight of the proposed equipment in kilograms",
                ),
            },
            required=["tower_id", "weight_kg"],
        ),
    ),
    types.FunctionDeclaration(
        name="check_regional_policy",
        description=(
            "Check if proposed equipment meets regional municipality rules for the "
            "tower's zone, including max mounting height and single-tenant weight cap."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "tower_id": types.Schema(
                    type=types.Type.STRING,
                    description="Tower identifier used to resolve the region",
                ),
                "height_m": types.Schema(
                    type=types.Type.NUMBER,
                    description="Proposed mounting height in meters",
                ),
                "weight_kg": types.Schema(
                    type=types.Type.NUMBER,
                    description="Weight of the proposed equipment in kilograms",
                ),
            },
            required=["tower_id", "height_m", "weight_kg"],
        ),
    ),
]


class LeaseVettingAgent:
    """Agent that vets lease requests using Google Gemini function calling."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        resolved_key = get_api_key(api_key)
        self._client = genai.Client(api_key=resolved_key)
        self._model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self._tools = types.Tool(function_declarations=TOOL_DEFINITIONS)
        self._config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[self._tools],
        )

    def vet_lease(self, request_text: str) -> dict[str, Any]:
        """Vet a plain-text lease request and return a structured judgment."""
        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part(text=request_text.strip())]
            )
        ]
        tool_results: list[dict[str, Any]] = []

        while True:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._config,
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts
            function_calls = [p for p in parts if p.function_call is not None]

            if not function_calls:
                break

            contents.append(types.Content(role="model", parts=parts))

            result_parts: list[types.Part] = []
            for part in function_calls:
                fc = part.function_call
                result = execute_tool(fc.name, dict(fc.args))
                tool_results.append(result)
                result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                )

            contents.append(types.Content(role="user", parts=result_parts))

        hard_error = self._detect_hard_error(tool_results)
        if hard_error:
            return self._build_hard_error_verdict(hard_error, tool_results)

        text = self._extract_text(response)
        verdict = self._parse_json(text)
        self._validate_verdict(verdict)
        return verdict

    def _detect_hard_error(self, tool_results: list[dict[str, Any]]) -> str | None:
        for result in tool_results:
            error = result.get("error")
            if error in (TOWER_NOT_FOUND, NO_POLICY_FOUND):
                return error
        return None

    def _build_hard_error_verdict(
        self, reason: str, tool_results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        tower_id = ""
        checks_run: list[dict[str, Any]] = []
        for result in tool_results:
            if result.get("tower_id"):
                tower_id = str(result["tower_id"])
            checks_run.append({
                "name": result.get("check_name", "unknown"),
                "passed": bool(result.get("passed")),
            })
        return {
            "status": "REJECTED",
            "reason": reason,
            "operator": "",
            "tower_id": tower_id,
            "checks_run": checks_run,
        }

    def _extract_text(self, response: Any) -> str:
        parts = response.candidates[0].content.parts
        text_parts = [p.text for p in parts if hasattr(p, "text") and p.text]
        if not text_parts:
            raise ValueError("Model response contained no text output.")
        return "\n".join(text_parts).strip()

    def _parse_json(self, text: str) -> dict[str, Any]:
        clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                raise ValueError(f"Could not parse JSON from model response: {text!r}")
            return json.loads(match.group())

    def _validate_verdict(self, verdict: dict[str, Any]) -> None:
        required = {"status", "reason", "operator", "tower_id", "checks_run"}
        missing = required - set(verdict.keys())
        if missing:
            raise ValueError(f"Verdict missing required fields: {missing}")
        if verdict["status"] not in ("APPROVED", "REJECTED"):
            raise ValueError(f"Invalid status: {verdict['status']!r}")
