import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model import (
    TechParams,
    empirical_regions,
    estimate_consumers,
    hyperplanes,
    load_data,
    observed_market_outcome,
    price_vertices,
    solve_scenario,
)

st.set_page_config(page_title="Information Provision Policy Lab", page_icon="☕", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
[data-testid="stMetric"] {border:1px solid #e5e7eb; border-radius:12px; padding:.8rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data("data.csv")

df = get_data()
s = 2.0  # M=2 technologies, K=1 qualifying label in this empirical adaptation.

st.title("Information Provision Policy Lab")
st.caption("Empirical two-tier implementation of Danilina & Grigoriev (2020) using the 11-week Regular/Premium coffee panel.")

with st.sidebar:
    st.header("Model inputs")
    st.metric("Implied stringency s", "2.0")
    margin = st.slider("Variable-cost share of observed price floor", 0.40, 0.90, 0.70, 0.05)
    phi_regular = st.number_input("Regular footprint φ₀", min_value=0.01, value=17.50, step=0.50)
    phi_premium = st.number_input("Premium footprint φ₁", min_value=0.01, value=3.50, step=0.50)
    gamma = st.number_input("Environmental weight γ", min_value=0.0, value=0.90, step=0.10)
    st.caption("Costs are calibrated from observed price floors; footprints are explicit inputs because the purchase panel has no direct environmental measurements.")

tech = TechParams(
    c0=margin * float(df.p_regular.min()),
    c1=margin * float(df.p_premium.min()),
    phi0=phi_regular,
    phi1=phi_premium,
    gamma=gamma,
)
consumers = estimate_consumers(df, s=s)

m = st.columns(4)
m[0].metric("Weeks", len(df))
m[1].metric("Households", 3)
m[2].metric("Regular price floor", f"{df.p_regular.min():.0f}")
m[3].metric("Premium price floor", f"{df.p_premium.min():.0f}")

with st.expander("How the 3-level model works", expanded=True):
    st.markdown("""
**Level 1 — Government:** selects the information/label design and, under mandatory certification, the policy prices.

**Level 2 — Industry / producers:** under CI and LI, prices are chosen to maximize aggregate industry profit subject to the feasible consumer regions. Under voluntary labelling, the producer's admissible disclosure decision is evaluated before the market outcome.

**Level 3 — Consumers:** each household chooses Regular, Premium, or Exit using the calibrated money-metric utility
**ωⱼₖ = bⱼₖ − pₖdⱼₖ + μⱼₖˢs**.

**Social welfare:** **W = Σⱼ ln(ωⱼ + 1) + Π − γΦ**, where Π is aggregate industry profit and Φ is the environmental footprint cost.

For K=1, the price space is two-dimensional. Bounding and indifference hyperplanes divide the (p₀,p₁) plane into demand regions; equilibrium candidates occur at the resulting vertices.
""")

# 1. Parameter estimation
st.header("1. Parameter Estimation")
rows = []
for c in consumers:
    rows.append({"Household": c.name, "d₀ Regular": c.d0, "d₁ Premium": c.d1,
                 "b₀": c.b0, "b₁": c.b1, "μ₀": c.mu0, "μ₁": c.mu1,
                 "Switching fit": f"{100*c.fit_accuracy:.0f}%"})
st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)
st.caption("Budgets are anchored at maximum observed expenditure. Only μ₁−μ₀ is identified by binary switching, so μ₀ is normalized to zero.")

# 2. Geometry and partition
st.header("2. Inferred Price-Space Partitioning & Vertices")
st.markdown("Applying geometric price-space partitioning to the calibrated bounding and indifference hyperplanes dissects the (p₀,p₁) plane into empirical demand regions.")

hp = hyperplanes(consumers, s)
vertices = price_vertices(consumers, s)

if not vertices.empty:
    fig = px.scatter(vertices, x="p_regular", y="p_premium", hover_data=["hyperplane_1", "hyperplane_2"],
                     labels={"p_regular":"p₀ (Regular Price)", "p_premium":"p₁ (Premium Price)"},
                     title="Inferred price-space vertices")
    obs = df[["p_regular", "p_premium"]].drop_duplicates()
    fig.add_scatter(x=obs.p_regular, y=obs.p_premium, mode="markers", name="Observed prices", marker_symbol="x", marker_size=10)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bounding & indifference hyperplanes")
    st.dataframe(hp[["kind", "household", "equation"]], use_container_width=True, hide_index=True)

    st.subheader("Empirical demand-region interpretation")
    st.dataframe(empirical_regions(df), use_container_width=True, hide_index=True)

    st.info("The region thresholds shown here (e.g. p₁≈75 and p₁≈81) are empirical switch thresholds from the observed panel. They should not be presented as exact structural parameters of the paper.")
else:
    st.warning("No non-negative price vertices were generated from the current calibration.")

# Explicit region cards matching the requested interpretation
region_cols = st.columns(3)
region_cols[0].markdown("**A · High Premium Price**\n\np₁ > 81 → H1 + H2 buy Regular; H3 exits.\n\nObserved weeks: 7, 8, 9.")
region_cols[1].markdown("**B · Intermediate**\n\n80 ≤ p₁ ≤ 81 and p₀ ≤ 55 → H1 + H2 Regular; H3 Premium.\n\nObserved weeks: 2, 6, 11.")
region_cols[2].markdown("**C · Low Premium Price**\n\np₁ ≤ 75 → Premium adoption rises; H3 remains active and H1 can switch at sufficiently high p₀.\n\nObserved weeks: 1, 4, 5, 10.")

# 3. Scenario equilibrium
st.header("3. Equilibrium Analysis Across Policy Scenarios")
scenario_names = {
    "C": "Mandatory Certification",
    "L": "Voluntary Labelling",
    "CI": "Certification + Industry Pricing",
    "LI": "Voluntary Labelling + Industry Pricing",
}

results = []
for code, label in scenario_names.items():
    try:
        r = solve_scenario(code, consumers, tech, s=s)
        results.append({
            "Scenario": f"{code} — {label}",
            "p₀": r["p_regular"], "p₁": r["p_premium"],
            "Regular demand": r["demand_regular"], "Premium demand": r["demand_premium"],
            "Coverage": sum(v != "Exit" for v in r["choices"].values()),
            "Industry profit Π": r["profit"], "Welfare W": r["welfare"],
            "Footprint Φ": r["footprint"], "Stringency s": r["stringency"],
        })
    except Exception as exc:
        results.append({"Scenario": f"{code} — {label}", "Error": str(exc)})

scenario_df = pd.DataFrame(results)
st.dataframe(scenario_df.round(2), use_container_width=True, hide_index=True)

st.subheader("Scenario interpretation")
st.markdown("""
- **C / CI (Mandatory):** the clean technology is tied to the qualifying Premium label. The government/industry equilibrium is evaluated over the feasible price vertices, so market coverage and environmental conversion are visible rather than inferred from a single observed week.
- **L / LI (Voluntary):** the producer can choose whether the clean technology is disclosed through the qualifying label. This can preserve higher Premium markups, but may also leave the green segment priced out when p₁ becomes too high.
- The exact numerical equilibrium is calibration-dependent because costs and footprints are not directly observed in the supplied panel.
""")

# 4. Historical validation
st.header("4. Empirical Comparison: Observed vs. C and CI")
observed = observed_market_outcome(df, consumers, tech, s=s)
c = next((r for r in results if r["Scenario"].startswith("C —") and "p₀" in r), None)
ci = next((r for r in results if r["Scenario"].startswith("CI —") and "p₀" in r), None)

summary = df[["T", "p_regular", "p_premium"]].rename(columns={"T":"Week", "p_regular":"Observed p₀", "p_premium":"Observed p₁"}).copy()
if c and ci:
    summary["C p₀"] = c["p₀"]; summary["C p₁"] = c["p₁"]
    summary["CI p₀"] = ci["p₀"]; summary["CI p₁"] = ci["p₁"]
    summary["Distance → C"] = np.hypot(summary["Observed p₀"]-c["p₀"], summary["Observed p₁"]-c["p₁"])
    summary["Distance → CI"] = np.hypot(summary["Observed p₀"]-ci["p₀"], summary["Observed p₁"]-ci["p₁"])
st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

if c and ci:
    obs_profit = observed.profit.mean(); obs_welfare = observed.welfare.mean()
    comparison = pd.DataFrame([
        ["Observed historical", df.p_regular.mean(), df.p_premium.mean(), obs_profit, obs_welfare],
        ["Scenario C", c["p₀"], c["p₁"], c["Industry profit Π"], c["Welfare W"]],
        ["Scenario CI", ci["p₀"], ci["p₁"], ci["Industry profit Π"], ci["Welfare W"]],
    ], columns=["Outcome", "Regular price", "Premium price", "Industry profit", "Social welfare"])
    st.subheader("Predicted vs. observed summary")
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

    plot_df = comparison.melt(id_vars="Outcome", value_vars=["Regular price","Premium price"], var_name="Type", value_name="Price")
    fig2 = px.bar(plot_df, x="Outcome", y="Price", color="Type", barmode="group", title="Observed vs. Scenario C / CI prices")
    st.plotly_chart(fig2, use_container_width=True)

    st.metric("Mean observed→C price distance", f"{summary['Distance → C'].mean():.2f}")
    st.metric("Mean observed→CI price distance", f"{summary['Distance → CI'].mean():.2f}")

# 5. Limitations
st.header("5. Validation & Identification Notes")
st.markdown("""
**Observed:** prices, quantities, expenditures and switching behavior.

**Estimated:** household demand quantities, expenditure-based budget bounds and the identifiable premium-vs-regular μ difference.

**Calibrated rather than identified:** variable costs and environmental footprints. The dataset contains no direct production-cost or footprint observations, so the app exposes these assumptions in the sidebar instead of silently treating them as measured facts.

**Validation target:** the model should reproduce the observed demand partition (H1/H2 Regular, H3 Premium, and H3 exit at high p₁) and provide a transparent comparison between historical prices and the theoretical C/CI vertices.
""")

with st.expander("Raw 11-week purchase panel"):
    st.dataframe(df, use_container_width=True, hide_index=True)
