from pathlib import Path

import altair as alt
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



def validate_panel(df, consumers, s=2.0):
    """Validate observed choices against the calibrated consumer utility.

    Kept in app.py so the UI remains robust if Streamlit serves a cached
    model.py from an earlier deployment revision.
    """
    consumer_by_name = {c.name: c for c in consumers}
    rows = []
    for r in df.itertuples(index=False):
        for i in range(1, 4):
            c = consumer_by_name[f"Household {i}"]
            q0 = getattr(r, f"h{i}_regular")
            q1 = getattr(r, f"h{i}_premium")
            if q0 > 0 and q1 > 0:
                observed = "Both"
            elif q0 > 0:
                observed = "Regular"
            elif q1 > 0:
                observed = "Premium"
            else:
                observed = "Exit"
            u0 = c.b0 - r.p_regular * c.d0 + c.mu0 * s
            u1 = c.b1 - r.p_premium * c.d1 + c.mu1 * s
            predicted_code, predicted_utility = model_choice = (
                int(np.argmax([0.0, u0, u1])) - 1,
                float(max(0.0, u0, u1)),
            )
            predicted = {-1: "Exit", 0: "Regular", 1: "Premium"}[predicted_code]
            comparable = observed != "Both"
            rows.append({
                "Week": int(r.T),
                "Household": c.name,
                "Observed choice": observed,
                "Predicted choice": predicted,
                "u₀ Regular": u0,
                "u₁ Premium": u1,
                "Predicted utility": predicted_utility,
                "Comparable": "Yes" if comparable else "No — both products observed",
                "Match": bool(predicted == observed) if comparable else np.nan,
            })
    return pd.DataFrame(rows)

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
    .block-container {max-width: 1250px; padding-top: 1.5rem;}
    [data-testid="stMetric"] {border:1px solid #e5e7eb; border-radius:12px; padding:.75rem;}
    .region-card {border:1px solid #e5e7eb; border-radius:12px; padding:14px; min-height:145px;}
    .legend-dot {font-size:1.15rem;}
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

# Two observed product types: Regular (k=0) and Premium (k=1).
s = 2.0

st.title("Information Provision Policy Lab")
st.caption(
    "Empirical two-tier implementation of Danilina & Grigoriev (2020) "
    "using the 11-week Regular/Premium coffee purchase panel."
)

with st.sidebar:
    st.header("Model inputs")
    st.info(
        "**How to read these values**\n\n"
        "The paper defines the theoretical model, but it does not provide "
        "the coffee-panel values below. Inputs marked as assumptions are "
        "therefore calibration choices, not observations from the paper.\n\n"
        "**Estimated from the panel:** household demand quantities, expenditure "
        "bounds and the identifiable relative μ component.\n\n"
        "**Calibrated/assumed:** production costs and environmental footprints. "
        "The dataset contains no direct cost or footprint measurements.\n\n"
        "**Policy calibration:** s = 2.0. The paper requires s ≥ 1 and derives "
        "stringency from the technology/label structure; this panel does not "
        "contain enough technology data to identify s directly.\n\n"
        "**Welfare calibration:** γ is the environmental-damage weight."
    )
    st.metric("Policy stringency s", "2.0")
    margin = st.slider(
        "Variable-cost share of observed price floor",
        0.40, 0.90, 0.70, 0.05,
        help="Calibration assumption: c_k is set as this share of the minimum observed price for product k. It is not estimated from the paper or directly observed in the panel.",
    )
    phi_regular = st.number_input(
        "Regular footprint φ₀", min_value=0.01, value=17.50, step=0.50,
        help="Calibration assumption. The purchase panel contains no lifecycle/environmental-footprint measurement.",
    )
    phi_premium = st.number_input(
        "Premium footprint φ₁", min_value=0.01, value=3.50, step=0.50,
        help="Calibration assumption. Lower than Regular here to represent the intended environmental ordering.",
    )
    gamma = st.number_input(
        "Environmental damage weight γ", min_value=0.0, value=0.90, step=0.10,
        help="Welfare parameter from the theoretical model; 0.90 is a calibration choice for this empirical exercise.",
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
**Level 1 — Government:** chooses the information/certification design and,
depending on the scenario, evaluates prices and welfare.

**Level 2 — Industry:** under CI/LI, industry pricing is explicitly
evaluated as a profit-maximization step.

**Level 3 — Consumers:** each household chooses Regular, Premium, or Exit
using the calibrated money-metric utility

$$\omega_{jk}=b_{jk}-p_kd_{jk}+\mu_{jk}^{s}s.$$

Social welfare is

$$W=\sum_j\ln(\omega_j+1)+\Pi-\gamma\Phi.$$

For two product types, price space is the $(p_0,p_1)$ plane. The paper
partitions this space using bounding and indifference hyperplanes and shows
that candidate optimal prices can be found at price vertices.
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

st.dataframe(pd.DataFrame(param_rows).round(2), use_container_width=True, hide_index=True)
st.caption(
    "Budgets are anchored at maximum observed expenditure. Only the relative "
    "Premium-vs-Regular μ component is identified by switching, so μ₀ is normalized to zero."
)

# ---------------------------------------------------------------------------
# 2. Price-space partitioning
# ---------------------------------------------------------------------------
st.header("2. Inferred Price-Space Partitioning & Vertices")
st.markdown(
    "This is the part of the app where the geometry should be visible. "
    "The lines are the actual calibrated bounding and indifference hyperplanes; "
    "their intersections are the model's price vertices. The observed weeks "
    "and scenario equilibria are overlaid so every point has a clear meaning."
)

hp = hyperplanes(consumers, s)
vertices = price_vertices(consumers, s)

# Build a readable plotting window around the observed data and model vertices.
all_x = list(df.p_regular.astype(float))
all_y = list(df.p_premium.astype(float))
if not vertices.empty:
    all_x += list(vertices.p_regular.astype(float))
    all_y += list(vertices.p_premium.astype(float))
x_min = 0.0
x_max = max(all_x) * 1.05
y_min = 0.0
y_max = max(all_y) * 1.05

# Hyperplanes -> explicit line segments in the plotting window.
line_rows = []
for idx, r in hp.iterrows():
    kind = r["kind"]
    if kind == "Bounding":
        if abs(r["a"]) > 1e-12:
            x = r["rhs"] / r["a"]
            if x_min <= x <= x_max:
                line_rows += [
                    {"line": f"B{idx+1} · {r['household']} · Regular affordability", "x": x, "y": y_min},
                    {"line": f"B{idx+1} · {r['household']} · Regular affordability", "x": x, "y": y_max},
                ]
        elif abs(r["b"]) > 1e-12:
            y = r["rhs"] / r["b"]
            if y_min <= y <= y_max:
                line_rows += [
                    {"line": f"B{idx+1} · {r['household']} · Premium affordability", "x": x_min, "y": y},
                    {"line": f"B{idx+1} · {r['household']} · Premium affordability", "x": x_max, "y": y},
                ]
    else:
        if abs(r["b"]) > 1e-12:
            xs = np.linspace(x_min, x_max, 120)
            ys = (r["rhs"] - r["a"] * xs) / r["b"]
            for x, y in zip(xs, ys):
                if y_min <= y <= y_max:
                    line_rows.append(
                        {"line": f"I{idx+1} · {r['household']} · Indifference", "x": float(x), "y": float(y)}
                    )

lines_df = pd.DataFrame(line_rows)

obs_plot = df[["T", "p_regular", "p_premium"]].copy()
obs_plot["Type"] = "Observed week"
obs_plot["Label"] = obs_plot["T"].map(lambda x: f"Week {int(x)}")

vertex_plot = pd.DataFrame()
if not vertices.empty:
    vertex_plot = vertices[["p_regular", "p_premium"]].copy()
    vertex_plot["Type"] = "Model vertex"
    vertex_plot["Label"] = [f"V{i+1}" for i in range(len(vertex_plot))]

st.subheader("2A. Bounding & indifference hyperplanes")
st.markdown(
    "**How to read it:** vertical lines are Regular affordability bounds; "
    "horizontal lines are Premium affordability bounds; diagonal lines are "
    "Regular-vs-Premium indifference boundaries. **A model vertex is where two "
    "or more of these boundaries intersect.**"
)
st.dataframe(
    hp[["kind", "household", "equation"]],
    use_container_width=True,
    hide_index=True,
)

# Compute scenarios before the main geometry chart so they can be shown on it.
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
valid_scenarios = scenario_df[scenario_df.get("p₀", pd.Series(dtype=float)).notna()].copy() if "p₀" in scenario_df else pd.DataFrame()

scenario_plot = pd.DataFrame()
if not valid_scenarios.empty:
    scenario_plot = valid_scenarios[["Code", "p₀", "p₁"]].copy()
    scenario_plot["Type"] = "Scenario equilibrium"
    scenario_plot["Label"] = scenario_plot["Code"]

# Layered Altair chart: boundaries + vertices + observed weeks + scenario points.
base = alt.Chart(pd.DataFrame({"x": [], "y": []})).properties(height=520)
if not lines_df.empty:
    line_chart = (
        alt.Chart(lines_df)
        .mark_line(opacity=0.35)
        .encode(
            x=alt.X("x:Q", title="Regular price p₀", scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("y:Q", title="Premium price p₁", scale=alt.Scale(domain=[y_min, y_max])),
            detail="line:N",
            tooltip=["line:N", alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("y:Q", format=".2f")],
        )
    )
else:
    line_chart = base

layers = [line_chart]

if not vertex_plot.empty:
    layers.append(
        alt.Chart(vertex_plot).mark_point(size=70, shape="diamond").encode(
            x="p_regular:Q", y="p_premium:Q",
            tooltip=[alt.Tooltip("Label:N"), alt.Tooltip("p_regular:Q", title="Regular"), alt.Tooltip("p_premium:Q", title="Premium")],
        )
    )
    layers.append(
        alt.Chart(vertex_plot).mark_text(dy=-10, fontSize=10).encode(
            x="p_regular:Q", y="p_premium:Q", text="Label:N"
        )
    )

layers.append(
    alt.Chart(obs_plot).mark_point(size=70, filled=True).encode(
        x="p_regular:Q", y="p_premium:Q",
        tooltip=[
            alt.Tooltip("Label:N"),
            alt.Tooltip("p_regular:Q", title="Regular price"),
            alt.Tooltip("p_premium:Q", title="Premium price"),
        ],
    )
)
layers.append(
    alt.Chart(obs_plot).mark_text(dy=10, fontSize=9).encode(
        x="p_regular:Q", y="p_premium:Q", text="Label:N"
    )
)

if not scenario_plot.empty:
    layers.append(
        alt.Chart(scenario_plot).mark_point(size=170, shape="star").encode(
            x="p₀:Q", y="p₁:Q",
            tooltip=[
                alt.Tooltip("Code:N", title="Scenario"),
                alt.Tooltip("p₀:Q", title="Regular equilibrium"),
                alt.Tooltip("p₁:Q", title="Premium equilibrium"),
            ],
        )
    )
    layers.append(
        alt.Chart(scenario_plot).mark_text(dy=-13, fontSize=12, fontWeight="bold").encode(
            x="p₀:Q", y="p₁:Q", text="Code:N"
        )
    )

st.altair_chart(alt.layer(*layers).interactive(), use_container_width=True)

st.caption(
    "Legend: ♦ = model price vertex; ● = observed weekly price combination; "
    "★ = scenario equilibrium. Hover over any line to see which household "
    "boundary generated it, and hover over V1, V2, etc. to inspect the vertex."
)

st.subheader("2B. Empirical demand regions")
st.info(
    "**Important distinction:** these are **market regions**, not separate "
    "regions for each household. One region represents **one combined pattern "
    "of consumer decisions across all three households** at the observed prices. "
    "Section 2B is empirically constructed from the panel: for every week, we "
    "translate each household's observed Regular/Premium quantities into a "
    "choice state (Regular, Premium, Both, or Exit), then group weeks that have "
    "the same joint household-choice signature. These regions are therefore "
    "descriptive empirical regions, not theoretical regions derived from the "
    "paper's hyperplanes."
)
region_df = empirical_regions(df)
st.dataframe(region_df, use_container_width=True, hide_index=True)

st.subheader("2C. Week-by-week model validation")
st.markdown(
    "This section is now a direct consequence of the calibrated utility model "
    "rather than a hand-written price threshold. For every observed week and "
    "household, the app evaluates Regular utility, Premium utility and Exit, "
    "then compares the model-implied choice with the observed choice."
)
validation_df = validate_panel(df, consumers, s=s)
comparable = validation_df[validation_df["Comparable"] == "Yes"].copy()
match_rate = float(comparable["Match"].mean()) if not comparable.empty else np.nan
v1, v2, v3 = st.columns(3)
v1.metric("Household-week observations", len(validation_df))
v2.metric("Comparable to single-choice model", len(comparable))
v3.metric("Model match rate", "N/A" if np.isnan(match_rate) else f"{100*match_rate:.1f}%")
st.caption(
    "Weeks where a household buys both products are shown explicitly as 'Both'. "
    "They are not counted as a literal match because the paper's consumer problem "
    "chooses one product type or Exit."
)
st.dataframe(validation_df.round(2), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 3. Four scenarios
# ---------------------------------------------------------------------------
st.header("3. Equilibrium Analysis Across Policy Scenarios")
st.markdown(
    "Each scenario is a **point in the same (p₀, p₁) price space**. "
    "The table and chart below therefore report both the Regular and Premium "
    "equilibrium prices for every scenario."
)
st.dataframe(scenario_df.round(2), use_container_width=True, hide_index=True)

if not scenario_plot.empty:
    st.subheader("3A. Scenario equilibria in price space")
    st.markdown(
        "The ★ markers are the four scenario outcomes: **C**, **L**, **CI**, "
        "and **LI**. Their coordinates are the model's equilibrium Regular "
        "and Premium prices under the current calibration."
    )
    st.altair_chart(
        alt.Chart(scenario_plot)
        .mark_point(size=220, shape="star")
        .encode(
            x=alt.X("p₀:Q", title="Regular equilibrium price p₀"),
            y=alt.Y("p₁:Q", title="Premium equilibrium price p₁"),
            text="Code:N",
            tooltip=[
                "Code:N",
                alt.Tooltip("p₀:Q", title="Regular"),
                alt.Tooltip("p₁:Q", title="Premium"),
            ],
        )
        .properties(height=420)
        .interactive(),
        use_container_width=True,
    )

st.subheader("3B. Scenario interpretation")
st.markdown(
    """
- **C — Mandatory Certification:** government selects the policy outcome
  directly under mandatory certification.
- **L — Voluntary Labelling:** producer/label responses are considered before
  the government compares welfare.
- **CI — Certification + Industry Pricing:** industry chooses the
  profit-maximizing price vertex given the certification structure.
- **LI — Voluntary Labelling + Industry Pricing:** industry pricing and
  voluntary disclosure are evaluated before the welfare comparison.
"""
)

# ---------------------------------------------------------------------------
# 4. Historical validation
# ---------------------------------------------------------------------------
st.header("4. Empirical Comparison: Observed vs. Scenario Equilibria")
observed = observed_market_outcome(df, consumers, tech, s=s)
c = next((r for r in results if r.get("Code") == "C" and "p₀" in r), None)
ci = next((r for r in results if r.get("Code") == "CI" and "p₀" in r), None)

summary = df[["T", "p_regular", "p_premium"]].rename(
    columns={"T": "Week", "p_regular": "Observed p₀", "p_premium": "Observed p₁"}
).copy()

if c and ci:
    summary["C p₀"] = c["p₀"]
    summary["C p₁"] = c["p₁"]
    summary["CI p₀"] = ci["p₀"]
    summary["CI p₁"] = ci["p₁"]
    summary["Distance → C"] = np.hypot(
        summary["Observed p₀"] - c["p₀"], summary["Observed p₁"] - c["p₁"]
    )
    summary["Distance → CI"] = np.hypot(
        summary["Observed p₀"] - ci["p₀"], summary["Observed p₁"] - ci["p₁"]
    )
    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)

    comparison = pd.DataFrame(
        [
            ["Observed historical", df.p_regular.mean(), df.p_premium.mean(), observed.profit.mean(), observed.welfare.mean()],
            ["Scenario C", c["p₀"], c["p₁"], c["Industry profit Π"], c["Welfare W"]],
            ["Scenario CI", ci["p₀"], ci["p₁"], ci["Industry profit Π"], ci["Welfare W"]],
        ],
        columns=["Outcome", "Regular price", "Premium price", "Industry profit", "Social welfare"],
    )
    st.subheader("4A. Predicted vs. observed summary")
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    a.metric("Mean observed → C price distance", f"{summary['Distance → C'].mean():.2f}")
    b.metric("Mean observed → CI price distance", f"{summary['Distance → CI'].mean():.2f}")

    st.info(
        "**Interpretation:** price distance measures market realism/fit; it does "
        "not mean the closest scenario is automatically the socially optimal one. "
        "Welfare and profit answer different questions."
    )

# ---------------------------------------------------------------------------
# 5. Market realism, profitability and welfare
# ---------------------------------------------------------------------------
st.header("5. Market Realism, Profitability & Social Welfare")

if not valid_scenarios.empty:
    realism_rows = []
    mean_p0 = float(df.p_regular.mean())
    mean_p1 = float(df.p_premium.mean())
    for _, r in valid_scenarios.iterrows():
        dist = float(np.hypot(r["p₀"] - mean_p0, r["p₁"] - mean_p1))
        realism_rows.append(
            {
                "Scenario": r["Code"],
                "Equilibrium Regular p₀": r["p₀"],
                "Equilibrium Premium p₁": r["p₁"],
                "Distance from observed mean": dist,
                "Industry profit Π": r["Industry profit Π"],
                "Social welfare W": r["Welfare W"],
            }
        )
    realism = pd.DataFrame(realism_rows).sort_values("Distance from observed mean")
    st.subheader("Market realism")
    st.markdown("**Which scenario resembles observed prices?**")
    st.dataframe(realism.round(2), use_container_width=True, hide_index=True)
    st.success(
        f"Closest to the observed mean price vector: **{realism.iloc[0]['Scenario']}** "
        f"(Euclidean distance {realism.iloc[0]['Distance from observed mean']:.2f})."
    )

    st.subheader("Profitability")
    best_profit = valid_scenarios.loc[valid_scenarios["Industry profit Π"].idxmax()]
    st.markdown("**Which produces the highest Π?**")
    st.metric("Highest industry profit", f"{best_profit['Code']} — Π = {best_profit['Industry profit Π']:.2f}")

    st.subheader("Social welfare")
    best_welfare = valid_scenarios.loc[valid_scenarios["Welfare W"].idxmax()]
    st.markdown("**Which produces the highest W?**")
    st.metric("Highest social welfare", f"{best_welfare['Code']} — W = {best_welfare['Welfare W']:.2f}")

    st.info(
        "These three rankings are deliberately kept separate: a scenario can be "
        "most realistic in price space, most profitable for industry, or best for "
        "social welfare. They need not be the same."
    )

# ---------------------------------------------------------------------------
# 6. Identification / limitations
# ---------------------------------------------------------------------------
st.header("6. Identification & Model-Input Notes")
st.markdown(
    """
**From the observed panel:** prices, quantities, expenditures and switching behavior.

**Estimated/calibrated from the panel:** household demand quantities, expenditure-based
budget bounds and the identifiable relative μ component.

**Not identified by this dataset:** production costs, environmental footprints,
technology delimiters and the underlying technology composition used to derive
policy stringency in the original framework. These are therefore explicit
calibration assumptions in this empirical adaptation.

**Important:** the theoretical paper supplies the model structure and solution
logic; the numerical coffee-panel values are part of this empirical adaptation,
not values reported by Danilina & Grigoriev (2020).
"""
)

with st.expander("Raw 11-week purchase panel"):
    st.dataframe(df, use_container_width=True, hide_index=True)
