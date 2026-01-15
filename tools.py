import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def _load_json(filename: str):
    path = DATA_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_crm_deal(deal_id: str) -> dict:
    deals = _load_json("crm_deals.json")
    for d in deals:
        if d["deal_id"].lower() == deal_id.lower():
            return d
    return {"error": f"Deal {deal_id} not found"}


def get_pricing_catalog() -> dict:
    return _load_json("pricing_catalog.json")


def get_template() -> dict:
    return _load_json("templates.json")


def search_past_proposals(industry: str) -> list:
    proposals = _load_json("past_proposals.json")
    return [p for p in proposals if p["industry"].lower() == industry.lower()]


def request_internal_approval(role: str, reason: str) -> dict:
    # Simulated Teams approval request
    return {
        "status": "requested",
        "role": role,
        "message": f"Approval requested from {role} for: {reason}"
    }
