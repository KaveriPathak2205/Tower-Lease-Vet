"""Tool definitions and implementations for the telecom tower lease vetting agent."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent
INVENTORY_PATH = DATA_DIR / "towers_inventory.json"
POLICIES_PATH = DATA_DIR / "regional_policies.txt"

TOWER_NOT_FOUND = "Tower not found in inventory."
NO_POLICY_FOUND = "No policy found for region."


@dataclass(frozen=True)
class TowerRecord:
    """A single tower entry from the inventory."""

    tower_id: str
    region: str
    max_allowed_weight_kg: float
    current_weight_kg: float


@dataclass(frozen=True)
class PolicyRecord:
    """Regional policy limits for equipment mounting."""

    region: str
    max_height_m: float
    weight_cap_kg: float
    notes: str = ""


@lru_cache(maxsize=1)
def _load_inventory() -> dict[str, TowerRecord]:
    """Load and cache the tower inventory keyed by tower_id."""
    with INVENTORY_PATH.open(encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)
    return {
        item["tower_id"]: TowerRecord(
            tower_id=item["tower_id"],
            region=item["region"],
            max_allowed_weight_kg=float(item["max_allowed_weight_kg"]),
            current_weight_kg=float(item["current_weight_kg"]),
        )
        for item in raw
    }


@lru_cache(maxsize=1)
def _load_policies() -> dict[str, PolicyRecord]:
    """Load and cache regional policies keyed by region name."""
    text = POLICIES_PATH.read_text(encoding="utf-8")
    blocks = re.split(r"\n\s*\n", text.strip())
    policies: dict[str, PolicyRecord] = {}

    for block in blocks:
        if not block.strip():
            continue
        region_match = re.search(r"^REGION:\s*(.+)$", block, re.MULTILINE)
        height_match = re.search(r"Max Equipment Height:\s*([\d.]+)\s*m", block)
        weight_match = re.search(
            r"Single-Tenant Weight Cap:\s*([\d.]+)\s*kg", block
        )
        notes_match = re.search(r"Notes:\s*(.+)$", block, re.MULTILINE)

        if not region_match or not height_match or not weight_match:
            continue

        region = region_match.group(1).strip()
        policies[region] = PolicyRecord(
            region=region,
            max_height_m=float(height_match.group(1)),
            weight_cap_kg=float(weight_match.group(1)),
            notes=notes_match.group(1).strip() if notes_match else "",
        )

    return policies


def check_tower_capacity(tower_id: str, weight_kg: float) -> dict[str, Any]:
    """
    Check whether adding equipment weight would exceed the tower's capacity.

    Args:
        tower_id: Identifier of the target tower (e.g. TWR-101).
        weight_kg: Weight of the proposed equipment in kilograms.

    Returns:
        Dictionary with check result and capacity details.
    """
    inventory = _load_inventory()
    tower = inventory.get(tower_id)

    if tower is None:
        return {
            "check_name": "check_tower_capacity",
            "passed": False,
            "error": TOWER_NOT_FOUND,
            "detail": TOWER_NOT_FOUND,
            "tower_id": tower_id,
        }

    projected = tower.current_weight_kg + weight_kg
    passed = projected <= tower.max_allowed_weight_kg
    remaining = tower.max_allowed_weight_kg - tower.current_weight_kg

    if passed:
        detail = (
            f"Tower {tower_id} has {remaining:.1f} kg remaining capacity; "
            f"adding {weight_kg} kg yields {projected:.1f} kg "
            f"(max {tower.max_allowed_weight_kg} kg)."
        )
    else:
        detail = (
            f"Tower {tower_id} would exceed capacity: "
            f"{projected:.1f} kg projected vs {tower.max_allowed_weight_kg} kg max "
            f"(only {remaining:.1f} kg available)."
        )

    return {
        "check_name": "check_tower_capacity",
        "passed": passed,
        "detail": detail,
        "tower_id": tower_id,
        "region": tower.region,
        "current_weight_kg": tower.current_weight_kg,
        "max_allowed_weight_kg": tower.max_allowed_weight_kg,
        "requested_weight_kg": weight_kg,
        "remaining_capacity_kg": remaining,
    }


def check_regional_policy(
    tower_id: str, height_m: float, weight_kg: float
) -> dict[str, Any]:
    """
    Check whether proposed equipment meets regional height and weight limits.

    Args:
        tower_id: Identifier of the target tower (region is resolved from inventory).
        height_m: Proposed mounting height in meters.
        weight_kg: Weight of the proposed equipment in kilograms.

    Returns:
        Dictionary with check result and policy details.
    """
    inventory = _load_inventory()
    tower = inventory.get(tower_id)

    if tower is None:
        return {
            "check_name": "check_regional_policy",
            "passed": False,
            "error": TOWER_NOT_FOUND,
            "detail": TOWER_NOT_FOUND,
            "tower_id": tower_id,
        }

    policies = _load_policies()
    policy = policies.get(tower.region)

    if policy is None:
        return {
            "check_name": "check_regional_policy",
            "passed": False,
            "error": NO_POLICY_FOUND,
            "detail": NO_POLICY_FOUND,
            "tower_id": tower_id,
            "region": tower.region,
        }

    height_ok = height_m <= policy.max_height_m
    weight_ok = weight_kg <= policy.weight_cap_kg
    passed = height_ok and weight_ok

    failures: list[str] = []
    if not height_ok:
        failures.append(
            f"height {height_m} m exceeds regional max {policy.max_height_m} m"
        )
    if not weight_ok:
        failures.append(
            f"weight {weight_kg} kg exceeds per-asset cap {policy.weight_cap_kg} kg"
        )

    if passed:
        detail = (
            f"Region {tower.region} allows up to {policy.max_height_m} m height "
            f"and {policy.weight_cap_kg} kg per asset; request is compliant."
        )
    else:
        detail = f"Region {tower.region} policy violation: {'; '.join(failures)}."

    return {
        "check_name": "check_regional_policy",
        "passed": passed,
        "detail": detail,
        "tower_id": tower_id,
        "region": tower.region,
        "max_height_m": policy.max_height_m,
        "weight_cap_kg": policy.weight_cap_kg,
        "requested_height_m": height_m,
        "requested_weight_kg": weight_kg,
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "check_tower_capacity",
        "description": (
            "Check if a tower has sufficient remaining weight capacity for new equipment. "
            "Looks up the tower in the inventory and verifies that "
            "current_weight + new_weight <= max_allowed_weight."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tower_id": {
                    "type": "string",
                    "description": "Tower identifier, e.g. TWR-101",
                },
                "weight_kg": {
                    "type": "number",
                    "description": "Weight of the proposed equipment in kilograms",
                },
            },
            "required": ["tower_id", "weight_kg"],
        },
    },
    {
        "name": "check_regional_policy",
        "description": (
            "Check if proposed equipment meets regional municipality rules for the "
            "tower's zone, including max mounting height and single-tenant weight cap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "tower_id": {
                    "type": "string",
                    "description": "Tower identifier used to resolve the region",
                },
                "height_m": {
                    "type": "number",
                    "description": "Proposed mounting height in meters",
                },
                "weight_kg": {
                    "type": "number",
                    "description": "Weight of the proposed equipment in kilograms",
                },
            },
            "required": ["tower_id", "height_m", "weight_kg"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Dispatch a tool call by name to the appropriate implementation.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Arguments provided by the model.

    Returns:
        JSON-serializable result dictionary.

    Raises:
        ValueError: If the tool name is not recognized.
    """
    if tool_name == "check_tower_capacity":
        return check_tower_capacity(
            tower_id=str(tool_input["tower_id"]),
            weight_kg=float(tool_input["weight_kg"]),
        )
    if tool_name == "check_regional_policy":
        return check_regional_policy(
            tower_id=str(tool_input["tower_id"]),
            height_m=float(tool_input["height_m"]),
            weight_kg=float(tool_input["weight_kg"]),
        )
    raise ValueError(f"Unknown tool: {tool_name}")
