from pathlib import Path

import numpy as np
import pandas as pd
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

# Community Cloud executes from the repository root. Using __file__ makes the
# data path deterministic both locally and in deployment.
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data.csv"

st.set_page_config(
    page_title="Information Provision Policy Lab",
    page_icon="☕",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {
        border: 1px solid #e5e7eb; border-radius: 12px; padding: .75rem;
    }
    .region-card {
        border: 1px solid #e5e7eb; border-radius: 12px;
        padding: 14px; min-height: 150px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Required dataset not found: {DATA_PATH}")
    data = load_data(str(DATA_PATH))
    required = {
        "T", "p_regular", "p_premium",
        "h1_regular", "h1_premium",
        "h2_regular", "h2_premium",
        "h3_regular", "h3_premium",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"data.csv is missing columns: {', '.join(missing)}")
    return data


try:
    df = get_data()
except Exception as exc:
    st.error("The application could not initialize the coffee panel.")
    st.exception(exc)
    st.stop()

# M=2 technologies and K=1 qualifying label.
s = 2.0

st.title("Information Provision Policy Lab")
st.caption(
    "Empirical two-tier implementation of Danilina & Grigoriev (2020) "
    "using the 11-week Regular/Premium coffee purchase panel."
)

with st.sidebar:
    st.header("Model inputs")
    st.metric("Implied stringency s", "2.0")
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
        "Environmental weight γ", min_value=0.0, value=0.90, step=0.10
    )
    st.caption(
        "Costs are calibrated from observed price floors. Footprints are "
        "explicit inputs because the purchase panel has no direct footprint data."
    )

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
    st.markdown(
        """
**Level 1 — Government:** chooses the information / certification design
and, in the mandatory case, the policy prices.

**Level 2 — Industry:** under CI/LI, chooses prices to maximize aggregate
profit; under voluntary labelling, the producer disclosure response is
evaluated before consumer demand.

**Level 3 — Consumers:** each household chooses Regular, Premium, or Exit
using the calibrated money-metric utility

$$\omega_{jk}=b_{jk}-p_kd_{jk}+\mu_{jk}^{s}s.$$

Social welfare is

$$W=\sum_j\ln(\omega_j+1)+\Pi-\gamma\Phi.$$

For K=1, the price space is two-dimensional. Bounding and indifference
hyperplanes partition $(p_0,p_1)$ into demand regions; candidate equilibria
are evaluated at the resulting vertices.
"""
    )

# ---------------------------------------------------------------------------
# 1. Parameter estimation
# ---------------------------------------------------------------------------
st.header("1. Parameter Estimation")

param_rows = []
for c in consumers:
    param_rows.append(
        {
            "Household": c.name,
            "d₀ Regular": c.d0,
            "d₁ Premium": c.d1,
            "b₀": c.b0,
            "b₁": c.b1,
            "μ₀": c.mu0,
            "μ₁": c.mu1,
            "Switching fit": f"{100*c.fit_accuracy:.0f}%",
        }
    )

st.dataframe(
    pd.DataFrame(param_rows).round(2),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "Budgets are anchored at maximum observed expenditure. Only μ₁−μ₀ "
    "is identified by binary switching, so μ₀ is normalized to zero."
)

# ---------------------------------------------------------------------------
# 2. Price-space partitioning
# ---------------------------------------------------------------------------
st.header("2. Inferred Price-Space Partitioning & Vertices")
st.markdown(
    "The calibrated bounding and indifference hyperplanes partition the "
    "$(p_0,p_1)$ plane. The chart overlays theoretical vertices and the "
    "11 observed weekly price combinations."
)

hp = hyperplanes(consumers, s)
vertices = price_vertices(consumers, s)

if vertices.empty:
    st.warning("No non-negative price vertices were generated under the current calibration.")
else:
    chart_vertices = vertices[["p_regular", "p_premium"]].copy()
    chart_vertices.columns = ["Regular price", "Premium price"]
    chart_observed = df[["p_regular", "p_premium"]].copy()
    chart_observed.columns = ["Regular price", "Premium price"]

    st.caption("Model vertices")
    st.scatter_chart(
        chart_vertices,
        x="Regular price",
        y="Premium price",
        height=420,
    )
    st.caption("Observed weekly price combinations")
    st.dataframe(
        chart_observed.assign(
            **{
                "Empirical threshold": np.where(
                    chart_observed["Premium price"] > 81,
                    "A — High Premium / H3 Exit",
                    np.where(
                        (chart_observed["Premium price"] >= 80)
                        & (chart_observed["Premium price"] <= 81)
                        & (chart_observed["Regular price"] <= 55),
                        "B — Intermediate / H3 Premium",
                        np.where(
                            chart_observed["Premium price"] <= 75,
                            "C — Low Premium / Premium Adoption",
                            "Outside core regions",
                        ),
                    ),
                )
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("2A. Bounding & indifference hyperplanes")
st.dataframe(
    hp[["kind", "household", "equation"]],
    use_container_width=True,
    hide_index=True,
)

st.subheader("2B. Empirical demand regions")
st.caption(
    "These are inferred switch regions from the observed panel, not claimed "
    "to be exact structural parameters of the theoretical paper."
)
region_df = empirical_regions(df)
st.dataframe(region_df, use_container_width=True, hide_index=True)

cols = st.columns(3)
region_cards = [
    (
        "A · High Premium Price",
        "p₁ > 81",
        "H1 + H2 buy Regular; H3 exits.",
        "Observed weeks: 7, 8, 9",
    ),
    (
        "B · Intermediate Premium Price",
        "80 ≤ p₁ ≤ 81 and p₀ ≤ 55",
        "H1 + H2 buy Regular; H3 buys Premium.",
        "Observed weeks: 2, 6, 11",
    ),
    (
        "C · Low Premium Price",
        "p₁ ≤ 75",
        "Premium adoption rises; H3 remains active.",
        "Observed weeks: 1, 4, 5, 10",
    ),
]
for col, (title, condition, behavior, weeks) in zip(cols, region_cards):
    col.markdown(
        f'<div class="region-card"><b>{title}</b><br><br>'
        f'<b>Condition:</b> {condition}<br>'
        f'<b>Behavior:</b> {behavior}<br>'
        f'<b>{weeks}</b></div>',
        unsafe_allow_html=True,
    )

st.subheader("2C. Week-by-week empirical classification")
def classify_region(row):
    if row.p_premium > 81:
        return "A — High Premium / H3 Exit"
    if 80 <= row.p_premium <= 81 and row.p_regular <= 55:
        return "B — Intermediate / H3 Premium"
    if row.p_premium <= 75:
        return "C — Low Premium / Premium Adoption"
    return "Outside the three core empirical regions"

weekly_regions = df[["T", "p_regular", "p_premium"]].copy()
weekly_regions["Region"] = weekly_regions.apply(classify_region, axis=1)
weekly_regions.columns = ["Week", "p₀ Regular", "p₁ Premium", "Empirical region"]
st.dataframe(
    weekly_regions,
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# 3. Four scenarios
# ---------------------------------------------------------------------------
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
        results.append(
            {
                "Code": code,
                "Scenario": label,
                "p₀": r["p_regular"],
                "p₁": r["p_premium"],
                "Regular demand": r["demand_regular"],
                "Premium demand": r["demand_premium"],
                "Coverage": sum(v != "Exit" for v in r["choices"].values()),
                "Industry profit Π": r["profit"],
                "Welfare W": r["welfare"],
                "Footprint Φ": r["footprint"],
                "Stringency s": r["stringency"],
            }
        )
    except Exception as exc:
        results.append({"Code": code, "Scenario": label, "Error": str(exc)})

scenario_df = pd.DataFrame(results)
st.dataframe(
    scenario_df.round(2),
    use_container_width=True,
    hide_index=True,
)

st.subheader("3A. Scenario interpretation")
st.markdown(
    """
- **C — Mandatory Certification:** government welfare objective evaluated
  under the mandatory clean-label assignment.
- **L — Voluntary Labelling:** government anticipates the producer's
  disclosure response before evaluating welfare.
- **CI — Certification + Industry Pricing:** industry selects the
  profit-maximizing feasible price vertex.
- **LI — Voluntary Labelling + Industry Pricing:** industry pricing is
  evaluated for each admissible voluntary disclosure mapping, followed by
  the welfare comparison.
"""
)

# ---------------------------------------------------------------------------
# 4. Historical validation
# ---------------------------------------------------------------------------
st.header("4. Empirical Comparison: Observed vs. C and CI")

observed = observed_market_outcome(df, consumers, tech, s=s)
c = next((r for r in results if r.get("Code") == "C" and "p₀" in r), None)
ci = next((r for r in results if r.get("Code") == "CI" and "p₀" in r), None)

summary = df[["T", "p_regular", "p_premium"]].rename(
    columns={
        "T": "Week",
        "p_regular": "Observed p₀",
        "p_premium": "Observed p₁",
    }
).copy()

if c and ci:
    summary["C p₀"] = c["p₀"]
    summary["C p₁"] = c["p₁"]
    summary["CI p₀"] = ci["p₀"]
    summary["CI p₁"] = ci["p₁"]
    summary["Distance → C"] = np.hypot(
        summary["Observed p₀"] - c["p₀"],
        summary["Observed p₁"] - c["p₁"],
    )
    summary["Distance → CI"] = np.hypot(
        summary["Observed p₀"] - ci["p₀"],
        summary["Observed p₁"] - ci["p₁"],
    )

st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

if c and ci:
    comparison = pd.DataFrame(
        [
            [
                "Observed historical",
                df.p_regular.mean(),
                df.p_premium.mean(),
                observed.profit.mean(),
                observed.welfare.mean(),
            ],
            [
                "Scenario C",
                c["p₀"],
                c["p₁"],
                c["Industry profit Π"],
                c["Welfare W"],
            ],
            [
                "Scenario CI",
                ci["p₀"],
                ci["p₁"],
                ci["Industry profit Π"],
                ci["Welfare W"],
            ],
        ],
        columns=[
            "Outcome",
            "Regular price",
            "Premium price",
            "Industry profit",
            "Social welfare",
        ],
    )

    st.subheader("4A. Predicted vs. observed summary")
    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        hide_index=True,
    )

    chart_comparison = comparison.set_index("Outcome")[["Regular price", "Premium price"]]
    st.bar_chart(chart_comparison, height=350)


    a, b = st.columns(2)
    a.metric("Mean observed → C price distance", f"{summary['Distance → C'].mean():.2f}")
    b.metric("Mean observed → CI price distance", f"{summary['Distance → CI'].mean():.2f}")
else:
    st.warning(
        "C/CI results could not be produced under the current calibration. "
        "Check the scenario error column above."
    )

# ---------------------------------------------------------------------------
# 5. Key takeaway
# ---------------------------------------------------------------------------
st.header("5. Key Takeaway")
st.markdown(
    """
**The empirical panel illustrates a clear price-space mechanism:**

1. When **p₁ > 81**, Household 3 exits while Households 1–2 remain on Regular.
2. In the **80–81 intermediate band**, Household 3 can remain on Premium.
3. At **p₁ ≤ 75**, Premium adoption expands materially.

Therefore, voluntary pricing can create a market-exclusion trade-off for the
lower-impact segment when the Premium price crosses its inferred threshold.
Certification changes the feasible policy space by constraining how the
environmental information is disclosed and priced.

The exact welfare-maximizing $(p_0,p_1)$ point is **computed by the model**
rather than hard-coded from the historical panel.
"""
)

# ---------------------------------------------------------------------------
# 6. Identification / limitations
# ---------------------------------------------------------------------------
st.header("6. Validation & Identification Notes")
st.markdown(
    """
**Observed:** prices, quantities, expenditures and switching behavior.

**Estimated:** household demand quantities, expenditure-based budget bounds,
and the identifiable Premium-vs-Regular μ difference.

**Calibrated rather than identified:** variable costs and environmental
footprints. The supplied panel contains no direct production-cost or
environmental-footprint measurements, so these assumptions remain explicit
sidebar inputs.

**Validation target:** reproduce the observed demand partition and compare
historical prices with the theoretical C/CI equilibrium candidates without
pretending that unobserved environmental or cost parameters were measured.
"""
)

with st.expander("Raw 11-week purchase panel"):
    st.dataframe(df, use_container_width=True, hide_index=True)
