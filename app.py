import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from model import (
    TechParams,
    estimate_consumers,
    load_data,
    observed_market_outcome,
    price_vertices,
    solve_scenario,
)

st.set_page_config(
    page_title="Information Provision Policy Lab",
    page_icon="☕",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
div[data-testid="stMetric"] {border: 1px solid #e5e7eb; padding: .8rem;
border-radius: 12px; background: white;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    return load_data()


df = get_data()

st.title("Information Provision Policy Lab")
st.caption(
    "Empirical two-tier implementation of Danilina & Grigoriev (2020) "
    "using the 11-week Regular/Premium coffee purchase panel."
)

with st.sidebar:
    st.header("Model inputs")
    st.markdown(
        "**Observed prices** come directly from the panel. "
        "Household quantities are the observed weekly purchases."
    )

    # With M=2 technologies and K=1 label, exactly one technology is labelled:
    # s = M / m_labelled = 2.
    s = 2.0
    st.metric("Implied stringency (s)", "2.0")

    margin = st.slider(
        "Variable-cost share of observed price floor",
        0.40, 0.90, 0.70, 0.05
    )
    phi_regular = st.number_input(
        "Regular footprint φ₀", min_value=0.01, value=17.50, step=0.50
    )
    phi_premium = st.number_input(
        "Premium footprint φ₁", min_value=0.01, value=3.50, step=0.50
    )
    gamma = st.number_input(
        "Environmental amplification γ", min_value=0.0, value=0.90, step=0.10
    )
    st.caption(
        "Default footprints are calibrated to 50% of the Regular price floor "
        "and 5% of the Premium price floor. The purchase panel contains no "
        "direct footprint measurements, so these remain explicit calibration inputs."
    )

p0_floor = float(df.p_regular.min())
p1_floor = float(df.p_premium.min())

tech = TechParams(
    c0=margin * p0_floor,
    c1=margin * p1_floor,
    phi0=phi_regular,
    phi1=phi_premium,
    gamma=gamma,
)
consumers = estimate_consumers(df, s=s)

cols = st.columns(4)
cols[0].metric("Weeks", len(df))
cols[1].metric("Households", 3)
cols[2].metric("Regular price floor", f"{p0_floor:.0f}")
cols[3].metric("Premium price floor", f"{p1_floor:.0f}")


with st.expander("How the model works — in plain English", expanded=True):
    st.markdown(r"""
**Level 1 — Government.** Chooses the environmental information design
(the delimiter between regular and premium technology). In Scenario C/L it
also controls prices; in CI/LI it anticipates the industry pricing response.

**Level 2 — Industry / producers.** In CI/LI, an industry association chooses
prices to maximize aggregate profit. In L/LI, producers can voluntarily use a
lower qualifying label but cannot greenwash.

**Level 3 — Consumers.** Each household chooses Regular, Premium, or no
purchase by maximizing the paper's money-metric utility
ω_jk = b_jk − p_k d_jk + μ_jk^s s.

**Welfare.** W = Σ_j ln(ω_j + 1) + Π − γΦ, where Π is industry profit and
Φ is the environmental footprint cost.

For K=1, the paper's price space is two-dimensional. Box hyperplanes and
consumer indifference hyperplanes create a finite set of price vertices.
This app enumerates those vertices and applies the paper's four scenario
logic to the empirical coffee panel.
""")


st.header("1. Parameter estimation")

param_rows = []
for c in consumers:
    param_rows.append({
        "Household": c.name,
        "d₀ regular": round(c.d0, 2),
        "d₁ premium": round(c.d1, 2),
        "b₀ budget": round(c.b0, 2),
        "b₁ budget": round(c.b1, 2),
        "μ₀": round(c.mu0, 2),
        "μ₁": round(c.mu1, 2),
        "Switching fit": f"{100*c.fit_accuracy:.0f}%",
    })

st.dataframe(
    pd.DataFrame(param_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Budgets are estimated as maximum observed expenditure for each "
    "household/type. Only μ₁−μ₀ is identified by binary switching, so μ₀ "
    "is normalized to zero. Switching fit is a diagnostic rather than a "
    "claim that the deterministic model perfectly explains every observation."
)

st.info(
    f"Technology calibration: c₀={tech.c0:.2f} and c₁={tech.c1:.2f}, "
    f"using {margin:.0%} of the observed price floors. Environmental footprints "
    "are explicit inputs because the purchase panel contains no footprint data."
)


st.header("2. Geometric price space")

vertices = price_vertices(consumers, s)

if not vertices.empty:
    fig = px.scatter(
        vertices,
        x="p_regular",
        y="p_premium",
        hover_data=["line_1", "line_2"],
        labels={
            "p_regular": "Regular price (p₀)",
            "p_premium": "Premium price (p₁)",
        },
        title="Price vertices from box and indifference hyperplanes",
    )
    observed_prices = df[["p_regular", "p_premium"]].drop_duplicates()
    fig.add_scatter(
        x=observed_prices.p_regular,
        y=observed_prices.p_premium,
        mode="markers",
        name="Observed weekly prices",
        marker=dict(symbol="x", size=9),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(vertices.round(2), use_container_width=True, hide_index=True)
else:
    st.warning("No price vertices were generated.")


st.header("3. Four policy scenarios")

scenario_names = {
    "C": "Certification",
    "L": "Voluntary Labelling",
    "CI": "Certification + Industry Pricing",
    "LI": "Voluntary Labelling + Industry Pricing",
}

results = []
for code, label in scenario_names.items():
    try:
        result = solve_scenario(code, consumers, tech, s=s)
        results.append({
            "Scenario": f"{code} — {label}",
            "p₀ Regular": result["p_regular"],
            "p₁ Premium": result["p_premium"],
            "Industry profit Π": result["profit"],
            "Welfare W": result["welfare"],
            "Stringency s": result["stringency"],
            "Footprint Φ": result["footprint"],
            "Regular demand": result["demand_regular"],
            "Premium demand": result["demand_premium"],
        })
    except Exception as exc:
        results.append({
            "Scenario": f"{code} — {label}",
            "Error": str(exc),
        })

scenario_df = pd.DataFrame(results)
st.dataframe(scenario_df.round(3), use_container_width=True, hide_index=True)


st.header("4. Historical validation")

observed = observed_market_outcome(df, consumers, tech, s=s)
summary = observed[
    ["week", "p_regular", "p_premium", "profit", "welfare", "footprint"]
].copy()
summary.columns = [
    "Week", "Observed p₀", "Observed p₁",
    "Model profit", "Model welfare", "Model footprint"
]

c_result = next(
    (r for r in results if r["Scenario"].startswith("C —")), None
)
ci_result = next(
    (r for r in results if r["Scenario"].startswith("CI —")), None
)

if (
    c_result and ci_result
    and "p₀ Regular" in c_result
    and "p₀ Regular" in ci_result
):
    summary["C p₀"] = c_result["p₀ Regular"]
    summary["C p₁"] = c_result["p₁ Premium"]
    summary["CI p₀"] = ci_result["p₀ Regular"]
    summary["CI p₁"] = ci_result["p₁ Premium"]

    summary["Distance observed→C"] = np.sqrt(
        (summary["Observed p₀"] - summary["C p₀"]) ** 2
        + (summary["Observed p₁"] - summary["C p₁"]) ** 2
    )
    summary["Distance observed→CI"] = np.sqrt(
        (summary["Observed p₀"] - summary["CI p₀"]) ** 2
        + (summary["Observed p₁"] - summary["CI p₁"]) ** 2
    )

    avg_c = summary["Distance observed→C"].mean()
    avg_ci = summary["Distance observed→CI"].mean()
    m1, m2 = st.columns(2)
    m1.metric("Mean price distance → C", f"{avg_c:.2f}")
    m2.metric("Mean price distance → CI", f"{avg_ci:.2f}")

st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

long = pd.DataFrame({
    "Series": ["Observed", "Scenario C", "Scenario CI"],
    "Regular": [
        df.p_regular.mean(),
        c_result["p₀ Regular"] if c_result else np.nan,
        ci_result["p₀ Regular"] if ci_result else np.nan,
    ],
    "Premium": [
        df.p_premium.mean(),
        c_result["p₁ Premium"] if c_result else np.nan,
        ci_result["p₁ Premium"] if ci_result else np.nan,
    ],
}).melt(id_vars="Series", var_name="Type", value_name="Price")

fig2 = px.bar(
    long,
    x="Series",
    y="Price",
    color="Type",
    barmode="group",
    title="Average observed price vs. C and CI equilibrium prices",
)
st.plotly_chart(fig2, use_container_width=True)


st.header("5. Interpretation & validation notes")

st.markdown("""
- **Directly observed:** weekly prices, quantities, expenditures and switching.
- **Estimated:** consumer demand quantities, budgets and the identifiable
  μ₁−μ₀ preference difference.
- **Not identified by this panel:** production costs and environmental
  footprints. Costs are anchored to observed price floors; footprints remain
  explicit calibration inputs.
- **Validation:** compare the theoretical C/CI price vertices with the
  historical price cloud and check whether model-implied consumer choices
  reproduce the observed Regular/Premium/no-purchase pattern.
- **Important limitation:** the panel has only 11 weeks and deterministic
  switching is not perfectly consistent for every household. The UI therefore
  exposes the switching-fit diagnostic instead of hiding this limitation.
""")

with st.expander("Raw purchase panel"):
    st.dataframe(df, use_container_width=True, hide_index=True)
