import streamlit as st
import pandas as pd
from lead_scoring import load_and_score

st.set_page_config(page_title="Lead Triage System", layout="wide")

st.title("Lead Qualification System")
st.caption(
    "Upload a lead export (CSV). The system cleans the data, filters out "
    "non-leads (job seekers, students, spam, vendors, VCs, competitors), "
    "scores real leads on fit + intent, and ranks them with a recommendation."
)

uploaded = st.file_uploader("Upload lead export CSV", type=["csv"])

default_path = "leads.csv"
use_sample = False
if not uploaded:
    use_sample = st.checkbox("Use the sample export instead", value=True)

df = None
if uploaded:
    df = load_and_score(uploaded)
elif use_sample:
    df = load_and_score(default_path)

if df is not None:
    counts = df["recommendation"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total rows processed", len(df))
    c2.metric("Contact Now", int(counts.get("CONTACT NOW", 0)))
    c3.metric("Nurture", int(counts.get("NURTURE", 0)))
    c4.metric("Disqualify", int(counts.get("DISQUALIFY", 0)))

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Contact Now", "Nurture", "All leads"])

    display_cols = [
        "lead_id", "company", "name", "email", "title", "employees_norm",
        "budget_norm", "total_score", "fit_score", "intent_score",
        "recommendation", "notes",
    ]
    rename = {
        "employees_norm": "employees (est.)",
        "budget_norm": "budget/mo (est.)",
    }

    with tab1:
        sub = df[df["recommendation"] == "CONTACT NOW"][display_cols].rename(columns=rename)
        st.dataframe(sub, use_container_width=True, hide_index=True)

    with tab2:
        sub = df[df["recommendation"] == "NURTURE"][display_cols].rename(columns=rename)
        st.dataframe(sub, use_container_width=True, hide_index=True)

    with tab3:
        sub = df[display_cols].rename(columns=rename)
        st.dataframe(sub, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Inspect a single lead's scoring breakdown")
    lead_options = df["lead_id"].tolist()
    chosen = st.selectbox("Pick a lead_id", lead_options)
    row = df[df["lead_id"] == chosen].iloc[0]

    colA, colB = st.columns(2)
    with colA:
        st.markdown(f"**{row['company']}** — {row['name']} ({row['title']})")
        st.write(row["notes"])
        st.markdown(f"**Recommendation: {row['recommendation']}**  (score: {row['total_score']}/100)")
    with colB:
        if row["non_lead_reason"] and isinstance(row["non_lead_reason"], str):
            st.warning(f"Filtered out as non-lead: {row['non_lead_reason']}")
        else:
            st.write("**Fit reasons:**")
            for r in row["fit_reasons"]:
                st.write(f"- {r}")
            st.write("**Intent reasons:**")
            for r in row["intent_reasons"]:
                st.write(f"- {r}")

    st.divider()
    csv_out = df.drop(columns=["_all_blank"]).to_csv(index=False)
    st.download_button("Download full scored list as CSV", csv_out, "scored_leads.csv", "text/csv")
else:
    st.info("Upload a CSV or check 'Use the sample export' to get started.")
