"""
Lead triage / qualification engine.

Takes a raw, messy lead export and produces a cleaned, scored, ranked list
with a recommendation per lead: CONTACT NOW / NURTURE / DISQUALIFY.

Designed to be reusable: point it at any future export with the same
rough column layout (lead_id, created, name, email, company, employees,
website, title, source, monthly_budget, notes) and it will re-clean,
re-score, and re-rank from scratch.
"""

import re
import pandas as pd
from dateutil import parser as dateparser


# ---------------------------------------------------------------------------
# 1. CLEANING HELPERS
# ---------------------------------------------------------------------------

def parse_date(value):
    """Handle the many date formats seen in the export. Returns pd.Timestamp or NaT."""
    if pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    try:
        # dayfirst=False because most formats here (06/28/2024) are clearly MM/DD
        return dateparser.parse(str(value).strip(), dayfirst=False, fuzzy=True)
    except (ValueError, OverflowError):
        return pd.NaT


def parse_employees(value):
    """
    Normalize employee-count strings into a single numeric estimate.
    Handles blanks, exact numbers, '~39', '70+', '35-55' ranges.
    Returns float or None.
    """
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s == "":
        return None
    s = s.replace("~", "").replace("+", "")
    if "-" in s:
        parts = s.split("-")
        try:
            nums = [float(p) for p in parts if p.strip() != ""]
            return sum(nums) / len(nums) if nums else None
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_budget(value):
    """
    Normalize messy monthly-budget strings into a monthly USD-equivalent number.
    Handles: blank, 'TBD', 'depends', '$6k/mo', '5,000/mo', '0', '500',
    '8k-12k', '$8,000/mo', '6-8k', '5k-7k', '10,000'.
    Returns float or None (None = unknown/unstated, NOT the same as 0).
    """
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in ("", "tbd", "depends", "n/a", "none"):
        return None

    s = s.replace("$", "").replace(",", "").replace("/mo", "").strip()

    # Range like "8k-12k" or "6-8k"
    if "-" in s:
        parts = s.split("-")
        nums = []
        for p in parts:
            p = p.strip()
            mult = 1000 if "k" in p else 1
            p = p.replace("k", "")
            try:
                nums.append(float(p) * mult)
            except ValueError:
                pass
        return sum(nums) / len(nums) if nums else None

    mult = 1000 if "k" in s else 1
    s = s.replace("k", "")
    try:
        return float(s) * mult
    except ValueError:
        return None


def normalize_lead_id(raw_id):
    """Strip 'L-' prefix and '-dup' suffix so duplicate submissions collapse together."""
    if pd.isna(raw_id):
        return None
    s = str(raw_id).strip().upper()
    s = s.replace("L-", "").replace("-DUP", "")
    return s


# ---------------------------------------------------------------------------
# 2. NON-LEAD DETECTION
# Patterns that mean "this isn't a real sales prospect at all" -- these get
# filtered out before scoring, not scored low. Matched against the notes field.
# ---------------------------------------------------------------------------

NON_LEAD_PATTERNS = {
    "job_seeker": [
        r"looking for a role", r"attaching my cv", r"hiring developers",
        r"join your team", r"i'd love to join",
    ],
    "student_academic": [
        r"cs student", r"final year student", r"university project",
        r"bootcamp grad", r"interview your founder", r"doing a project on ai",
    ],
    "journalist": [
        r"journalist writing about",
    ],
    "vc_investor": [
        r"vc here", r"portfolio companies",
    ],
    "spam_scam": [
        r"you have won", r"click here to claim", r"smm panel",
        r"buy followers", r"bulk email blasting", r"high-da backlinks",
        r"reply stop",
    ],
    "vendor_pitching_us": [
        r"offshore dev team", r"automation devs on our bench",
        r"place candidates",
    ],
    "competitor_recon": [
        r"researching the market", r"fellow agency owner",
        r"run a competing automation agency", r"curious about your pricing for benchmarking",
        r"we do similar work",
    ],
    "junk_row": [
        r"newsletter signup", r"qa test entry", r"ignore this",
        r"follow up later\?\?", r"broken email",
    ],
}

# lead_ids or names that are obviously seed/test data, not real leads
JUNK_ID_PATTERNS = [r"^9\d{3}$"]  # normalized ids like 9001-9004 (from L-9001 etc.)
JUNK_EXACT_VALUES = {"testrow", "asdf", "test", "header"}


def classify_non_lead(row):
    """Return a reason string if this row is not a real sales lead, else None."""
    lead_id_norm = str(row.get("lead_id_norm") or "").lower()
    name = str(row.get("name") or "").strip().lower()
    company = str(row.get("company") or "").strip().lower()
    notes = str(row.get("notes") or "").lower()

    if lead_id_norm in JUNK_EXACT_VALUES or name in JUNK_EXACT_VALUES or company in JUNK_EXACT_VALUES:
        return "junk_row"
    for pat in JUNK_ID_PATTERNS:
        if re.match(pat, lead_id_norm):
            return "junk_row"
    if notes.strip() == "" or row.get("_all_blank"):
        return "junk_row"

    for category, patterns in NON_LEAD_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, notes):
                return category
    return None


# ---------------------------------------------------------------------------
# 3. SCORING LOGIC (applied only to real leads)
# ---------------------------------------------------------------------------

AGENCY_ICP_PATTERN = r"we're an? .*(agency|gtm)"
ADJACENT_ICP_PATTERN = r"(ecom brand|saas company|car dealership|solo consultant)"

BUDGET_APPROVED_PATTERN = r"budget approved"
DECISION_AUTHORITY_PATTERNS = [
    r"i make the call here", r"decision is mine", r"this is my priority to solve",
]
URGENCY_PATTERNS = [
    r"asap", r"this month", r"next 2 weeks", r"in 2 weeks",
    r"keen to move fast", r"priority for the quarter", r"decision this month",
]
COMPARING_PATTERN = r"comparing a few options"
UNSURE_PATTERN = r"not totally sure what we need yet"
LOOP_IN_TEAM_PATTERN = r"would need to loop in the team"
DECISION_TIMELINE_PATTERN = r"decision in about a month"
PRICE_SENSITIVE_PATTERN = r"price sensitive"
BELOW_RANGE_PATTERN = r"budget way below range"
MAYBE_LATER_PATTERN = r"maybe later"
TINY_SHOP_PATTERN = r"tiny budget, one-man shop"
EARLY_STARTUP_PATTERN = r"no real budget yet but sharp and might grow"


def score_fit(row):
    """0-50 points: how well this lead matches the target customer profile."""
    notes = str(row.get("notes") or "").lower()
    score = 0
    reasons = []

    # ICP match
    if re.search(AGENCY_ICP_PATTERN, notes):
        score += 20
        reasons.append("Core ICP (agency), +20")
    elif re.search(ADJACENT_ICP_PATTERN, notes):
        score += 10
        reasons.append("Adjacent ICP (non-agency w/ automation need), +10")

    # Company size
    emp = row.get("employees_norm")
    if emp is not None:
        if emp >= 10:
            score += 15
            reasons.append(f"Company size {emp:.0f} (>=10), +15")
        elif emp >= 1:
            score += 5
            reasons.append(f"Company size {emp:.0f} (1-9), +5")

    # Budget size
    budget = row.get("budget_norm")
    if budget is not None:
        if budget >= 8000:
            score += 15
            reasons.append(f"Budget ${budget:,.0f}/mo (>=8k), +15")
        elif budget >= 4000:
            score += 10
            reasons.append(f"Budget ${budget:,.0f}/mo (4-8k), +10")
        elif budget > 0:
            score += 5
            reasons.append(f"Budget ${budget:,.0f}/mo (<4k), +5")

    # Explicit poor-fit signals cap the score low regardless of the above
    if re.search(BELOW_RANGE_PATTERN, notes) or re.search(TINY_SHOP_PATTERN, notes):
        score = min(score, 10)
        reasons.append("Explicit poor-fit signal, capped at 10")

    return score, reasons


def score_intent(row):
    """0-50 points: how ready/willing this lead is to buy now."""
    notes = str(row.get("notes") or "").lower()
    score = 0
    reasons = []

    if re.search(BUDGET_APPROVED_PATTERN, notes):
        score += 30
        reasons.append("Budget approved, +30")
        if any(re.search(p, notes) for p in DECISION_AUTHORITY_PATTERNS):
            score += 10
            reasons.append("Decision authority stated, +10")
        if any(re.search(p, notes) for p in URGENCY_PATTERNS):
            score += 10
            reasons.append("Urgency signal, +10")
    else:
        if re.search(COMPARING_PATTERN, notes):
            score += 15
            reasons.append("Actively comparing options, +15")
        elif re.search(UNSURE_PATTERN, notes):
            score += 8
            reasons.append("Unsure what's needed, +8")
        elif re.search(LOOP_IN_TEAM_PATTERN, notes):
            score += 8
            reasons.append("Needs to loop in team, +8")

        if re.search(DECISION_TIMELINE_PATTERN, notes):
            score += 10
            reasons.append("Has a decision timeline (~1 month), +10")
        if re.search(PRICE_SENSITIVE_PATTERN, notes):
            score -= 5
            reasons.append("Price sensitive, -5")
        if re.search(MAYBE_LATER_PATTERN, notes):
            score = min(score, 5)
            reasons.append("Explicitly 'maybe later', capped at 5")
        if re.search(EARLY_STARTUP_PATTERN, notes):
            score += 10
            reasons.append("Early-stage but promising, +10 (nurture candidate)")

    return max(score, 0), reasons


def recommend(total_score, non_lead_reason):
    if isinstance(non_lead_reason, str) and non_lead_reason:
        return "DISQUALIFY"
    if total_score >= 65:
        return "CONTACT NOW"
    if total_score >= 30:
        return "NURTURE"
    return "DISQUALIFY"


# ---------------------------------------------------------------------------
# 4. PIPELINE
# ---------------------------------------------------------------------------

def load_and_score(csv_path_or_buffer):
    df = pd.read_csv(csv_path_or_buffer, dtype=str, keep_default_na=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Flag fully-blank rows before anything else
    df["_all_blank"] = df.apply(
        lambda r: all(str(v).strip() == "" for v in r.values), axis=1
    )

    df["lead_id_norm"] = df["lead_id"].apply(normalize_lead_id)
    df["created_norm"] = df["created"].apply(parse_date)
    df["employees_norm"] = df["employees"].apply(parse_employees)
    df["budget_norm"] = df["monthly_budget"].apply(parse_budget) if "monthly_budget" in df.columns else None

    # Drop the stray embedded header row and fully blank rows outright
    df = df[df["lead_id"].str.lower() != "header"]
    df = df[~df["_all_blank"]]

    # Deduplicate: same normalized lead_id OR same email -> keep first occurrence
    df = df.drop_duplicates(subset=["lead_id_norm"], keep="first")
    df = df.drop_duplicates(subset=["email"], keep="first")

    # Classify + score
    non_lead_reasons = []
    fit_scores, fit_reasons = [], []
    intent_scores, intent_reasons = [], []

    for _, row in df.iterrows():
        reason = classify_non_lead(row)
        non_lead_reasons.append(reason)
        if reason:
            fit_scores.append(0)
            intent_scores.append(0)
            fit_reasons.append([])
            intent_reasons.append([])
        else:
            f, fr = score_fit(row)
            i, ir = score_intent(row)
            fit_scores.append(f)
            intent_scores.append(i)
            fit_reasons.append(fr)
            intent_reasons.append(ir)

    df["non_lead_reason"] = non_lead_reasons
    df["fit_score"] = fit_scores
    df["intent_score"] = intent_scores
    df["total_score"] = df["fit_score"] + df["intent_score"]
    df["fit_reasons"] = fit_reasons
    df["intent_reasons"] = intent_reasons
    df["recommendation"] = df.apply(
        lambda r: recommend(r["total_score"], r["non_lead_reason"]), axis=1
    )

    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    return df


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "leads.csv"
    result = load_and_score(path)
    print(result["recommendation"].value_counts())
    print(f"\nTotal rows processed: {len(result)}")
    print("\nTop 10:")
    print(result[["lead_id", "company", "total_score", "recommendation"]].head(10))
