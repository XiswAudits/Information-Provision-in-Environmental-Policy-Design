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
    "The paper's price-space method answers one question: **at which combinations "
    "of Regular and Premium prices can consumer choices change?** For two products, "
    "we work in a two-dimensional plane: x = Regular price (p₀), y = Premium price (p₁)."
)

hp = hyperplanes(consumers, s)
vertices = price_vertices(consumers, s)

# A stable, readable plotting window. Keep the axes tied to the observed market
# plus the relevant model boundaries rather than letting a few extreme vertices
# dominate the visual.
obs_x_max = float(df.p_regular.max())
obs_y_max = float(df.p_premium.max())
positive_vertices = vertices[(vertices.p_regular > 0) & (vertices.p_premium > 0)].copy()
x_max = max(obs_x_max, float(positive_vertices.p_regular.max()) if not positive_vertices.empty else 0.0) * 1.08
y_max = max(obs_y_max, float(positive_vertices.p_premium.max()) if not positive_vertices.empty else 0.0) * 1.08
x_min, y_min = 0.0, 0.0

st.subheader("2A. Bounding & indifference hyperplanes")
st.info(
    "**How to read the geometry**\n\n"
    "• **Vertical line = Regular affordability boundary.** To the left of it, "
    "that household can afford Regular; to the right, Regular gives negative utility "
    "and the household exits that option.\n\n"
    "• **Horizontal line = Premium affordability boundary.** Below it, Premium is "
    "affordable; above it, Premium gives negative utility.\n\n"
    "• **Diagonal line = Regular/Premium indifference boundary.** On this line, "
    "the household gets exactly the same utility from Regular and Premium. "
    "Crossing the line can change which product the model predicts.\n\n"
    "• **◆ Vertex = intersection of two boundaries.** These intersections are the "
    "candidate price points used by the paper's finite price-space solution.\n\n"
    "The shaded/segmented lines are therefore not demand curves. They are **choice "
    "boundaries in price space**."
)

st.latex(r"\text{Regular affordability: }\quad p_0=\frac{b_{j0}+\mu_{j0}^{s}s}{d_{j0}}")
st.latex(r"\text{Premium affordability: }\quad p_1=\frac{b_{j1}+\mu_{j1}^{s}s}{d_{j1}}")
st.latex(
    r"\text{Indifference: }\quad "
    r"p_0d_{j0}-p_1d_{j1}=(b_{j0}-b_{j1})+(\mu_{j0}^{s}-\mu_{j1}^{s})s"
)

# Give the user a concrete example rather than asking them to infer the meaning
# from a dense equation table.
if not positive_vertices.empty:
    example_v = positive_vertices.iloc[0]
    st.success(
        f"**Example:** {example_v['p_regular']:.2f} Regular and "
        f"{example_v['p_premium']:.2f} Premium is a model vertex because it is "
        f"the intersection of **{example_v['hyperplane_1']}** and "
        f"**{example_v['hyperplane_2']}**. At a vertex, multiple choice/affordability "
        "boundaries meet."
    )

# Build line segments with simple ASCII field names. This avoids Altair field
# parsing problems with Unicode subscripts and makes the scenario overlay robust.
line_rows = []
for idx, r in hp.iterrows():
    if r["kind"] == "Bounding":
        if abs(r["a"]) > 1e-12:
            x = float(r["rhs"] / r["a"])
            if x_min <= x <= x_max:
                line_rows += [
                    {"line_id": f"B{idx+1}", "x": x, "y": y_min,
                     "description": f"B{idx+1}: {r['household']} Regular affordability"},
                    {"line_id": f"B{idx+1}", "x": x, "y": y_max,
                     "description": f"B{idx+1}: {r['household']} Regular affordability"},
                ]
        elif abs(r["b"]) > 1e-12:
            y = float(r["rhs"] / r["b"])
            if y_min <= y <= y_max:
                line_rows += [
                    {"line_id": f"B{idx+1}", "x": x_min, "y": y,
                     "description": f"B{idx+1}: {r['household']} Premium affordability"},
                    {"line_id": f"B{idx+1}", "x": x_max, "y": y,
                     "description": f"B{idx+1}: {r['household']} Premium affordability"},
                ]
    else:
        if abs(r["b"]) > 1e-12:
            xs = np.linspace(x_min, x_max, 160)
            ys = (float(r["rhs"]) - float(r["a"]) * xs) / float(r["b"])
            for x, y in zip(xs, ys):
                if y_min <= y <= y_max:
                    line_rows.append({
                        "line_id": f"I{idx+1}",
                        "x": float(x),
                        "y": float(y),
                        "description": f"I{idx+1}: {r['household']} Regular/Premium indifference",
                    })

lines_df = pd.DataFrame(line_rows)

obs_plot = df[["T", "p_regular", "p_premium"]].copy()
obs_plot["label"] = obs_plot["T"].map(lambda x: f"W{int(x)}")

vertex_plot = positive_vertices[["p_regular", "p_premium"]].copy()
vertex_plot["label"] = [f"V{i+1}" for i in range(len(vertex_plot))]

# Compute scenario outcomes before plotting so Scenario 3A and Section 2 use
# exactly the same objects.
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
            "Code": code,
            "Scenario": label,
            "regular_price": float(r["p_regular"]),
            "premium_price": float(r["p_premium"]),
            "Regular demand": r["demand_regular"],
            "Premium demand": r["demand_premium"],
            "Coverage": sum(v != "Exit" for v in r["choices"].values()),
            "Industry profit": r["profit"],
            "Welfare": r["welfare"],
            "Footprint": r["footprint"],
            "Stringency": r["stringency"],
        })
    except Exception as exc:
        results.append({"Code": code, "Scenario": label, "Error": str(exc)})

scenario_df = pd.DataFrame(results)
if "regular_price" in scenario_df.columns:
    valid_scenarios = scenario_df[
        scenario_df["regular_price"].notna() & scenario_df["premium_price"].notna()
    ].copy()
else:
    valid_scenarios = pd.DataFrame()

if not valid_scenarios.empty:
    scenario_plot = valid_scenarios[["Code", "regular_price", "premium_price"]].copy()
    scenario_plot["label"] = scenario_plot["Code"]
else:
    scenario_plot = pd.DataFrame(columns=["Code", "regular_price", "premium_price", "label"])

st.markdown("#### Price-space map")
st.caption(
    "Think of this as a map: **x = Regular price, y = Premium price**. "
    "The observed weeks are market locations; ◆ vertices are model-generated "
    "candidate boundaries; ★ marks are scenario outcomes."
)

# Layer 1: theoretical boundaries.
chart_layers = []
if not lines_df.empty:
    chart_layers.append(
        alt.Chart(lines_df).mark_line(opacity=0.38).encode(
            x=alt.X("x:Q", title="Regular price p₀", scale=alt.Scale(domain=[x_min, x_max])),
            y=alt.Y("y:Q", title="Premium price p₁", scale=alt.Scale(domain=[y_min, y_max])),
            detail="line_id:N",
            tooltip=[
                alt.Tooltip("description:N", title="Boundary"),
                alt.Tooltip("x:Q", title="Regular", format=".2f"),
                alt.Tooltip("y:Q", title="Premium", format=".2f"),
            ],
        )
    )

# Layer 2: vertices.
if not vertex_plot.empty:
    chart_layers.append(
        alt.Chart(vertex_plot).mark_point(size=85, shape="diamond", filled=True).encode(
            x=alt.X("p_regular:Q", title="Regular price p₀"),
            y=alt.Y("p_premium:Q", title="Premium price p₁"),
            tooltip=[
                alt.Tooltip("label:N", title="Vertex"),
                alt.Tooltip("p_regular:Q", title="Regular", format=".2f"),
                alt.Tooltip("p_premium:Q", title="Premium", format=".2f"),
            ],
        )
    )
    chart_layers.append(
        alt.Chart(vertex_plot).mark_text(dy=-11, fontSize=10).encode(
            x="p_regular:Q", y="p_premium:Q", text="label:N"
        )
    )

# Layer 3: observed weekly prices.
chart_layers.append(
    alt.Chart(obs_plot).mark_point(size=70, filled=True).encode(
        x="p_regular:Q", y="p_premium:Q",
        tooltip=[
            alt.Tooltip("label:N", title="Observed week"),
            alt.Tooltip("p_regular:Q", title="Regular price", format=".2f"),
            alt.Tooltip("p_premium:Q", title="Premium price", format=".2f"),
        ],
    )
)
chart_layers.append(
    alt.Chart(obs_plot).mark_text(dy=10, fontSize=9).encode(
        x="p_regular:Q", y="p_premium:Q", text="label:N"
    )
)

# Layer 4: scenario outcomes. If a scenario fails, the table below will expose
# the actual error instead of silently producing an empty graph.
if not scenario_plot.empty:
    chart_layers.append(
        alt.Chart(scenario_plot).mark_point(size=210, shape="star", filled=True).encode(
            x="regular_price:Q", y="premium_price:Q",
            tooltip=[
                alt.Tooltip("Code:N", title="Scenario"),
                alt.Tooltip("regular_price:Q", title="Regular equilibrium", format=".2f"),
                alt.Tooltip("premium_price:Q", title="Premium equilibrium", format=".2f"),
            ],
        )
    )
    chart_layers.append(
        alt.Chart(scenario_plot).mark_text(dy=-14, fontSize=12, fontWeight="bold").encode(
            x="regular_price:Q", y="premium_price:Q", text="label:N"
        )
    )

if chart_layers:
    st.altair_chart(
        alt.layer(*chart_layers).properties(height=560).interactive(),
        use_container_width=True,
    )

st.caption(
    "Legend: ◆ model vertex | ● observed weekly price | ★ scenario equilibrium. "
    "Hover over a boundary to see its source household and meaning."
)

with st.expander("How do I interpret a vertex?", expanded=False):
    st.markdown(
        """
1. Pick a **◆ V-number**.
2. Read its **(Regular price, Premium price)** coordinates.
3. Hover over the point to see the two boundaries that intersect there.
4. Those boundaries are places where at least one household's predicted choice
   can change.
5. The paper's optimization then evaluates these candidate vertices rather than
   searching every possible continuous price combination.

**Important:** a vertex is a *candidate price point*, not automatically the
optimal price and not necessarily an observed market price.
"""
    )

st.subheader("2B. Empirical demand regions")
st.info(
    "**These are market regions, not household-specific regions.** One region "
    "represents **one combined pattern of decisions across all three households**. "
    "Section 2B is empirically constructed from the panel: each week is translated "
    "into a household choice state (Regular, Premium, Both, or Exit), and weeks "
    "with the same three-household signature are grouped together. These regions "
    "describe what was observed in the coffee market; they are not the theoretical "
    "hyperplane regions from the paper."
)
region_df = empirical_regions(df)
st.dataframe(region_df, use_container_width=True, hide_index=True)

st.subheader("2C. Week-by-week model validation")
st.markdown(
    "For every observed **week × household**, the calibrated utility model is "
    "evaluated at the observed prices. The app then compares the model-implied "
    "choice (Regular, Premium or Exit) with the observed purchase state."
)
validation_df = validate_panel(df, consumers, s=s)
comparable = validation_df[validation_df["Comparable"] == "Yes"].copy()
match_rate = float(comparable["Match"].mean()) if not comparable.empty else np.nan
v1, v2, v3 = st.columns(3)
v1.metric("Household-week observations", len(validation_df))
v2.metric("Comparable to single-choice model", len(comparable))
v3.metric("Model match rate", "N/A" if np.isnan(match_rate) else f"{100*match_rate:.1f}%")
st.caption(
    "A 'Both' observation is shown explicitly and excluded from literal accuracy "
    "because the paper's consumer problem chooses one product type or Exit."
)
st.dataframe(validation_df.round(2), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 3. Four scenarios
# ---------------------------------------------------------------------------
st.header("3. Equilibrium Analysis Across Policy Scenarios")
st.markdown(
    "The four scenarios are **four points in the same two-dimensional price "
    "space**. Each point has two coordinates: its Regular equilibrium price "
    "p₀ and Premium equilibrium price p₁."
)

st.dataframe(scenario_df.round(2), use_container_width=True, hide_index=True)

if not valid_scenarios.empty:
    st.subheader("3A. Scenario equilibria in price space")
    st.info(
        "**How to read this graph:** x-axis = Regular equilibrium price p₀; "
        "y-axis = Premium equilibrium price p₁. Each ★ is one scenario. "
        "For example, if ★ CI is at (45, 80), that means the CI scenario's "
        "equilibrium is Regular = 45 and Premium = 80. The four points can be "
        "compared directly with the observed weekly price points in Section 2."
    )
    st.altair_chart(
        alt.Chart(scenario_plot)
        .mark_point(size=260, shape="star", filled=True)
        .encode(
            x=alt.X("regular_price:Q", title="Regular equilibrium price p₀"),
            y=alt.Y("premium_price:Q", title="Premium equilibrium price p₁"),
            tooltip=[
                alt.Tooltip("Code:N", title="Scenario"),
                alt.Tooltip("regular_price:Q", title="Regular equilibrium", format=".2f"),
                alt.Tooltip("premium_price:Q", title="Premium equilibrium", format=".2f"),
            ],
        )
        .properties(height=450)
        .interactive(),
        use_container_width=True,
    )
else:
    st.error(
        "The scenario solver returned no valid equilibrium points. "
        "See the Scenario table above for the exact error returned by each scenario."
    )

st.subheader("3B. Scenario interpretation")
st.markdown(
    """
- **C — Mandatory Certification:** government selects the policy outcome directly under mandatory certification.
- **L — Voluntary Labelling:** producer/label responses are considered before the government compares welfare.
- **CI — Certification + Industry Pricing:** industry chooses the profit-maximizing price vertex given the certification structure.
- **LI — Voluntary Labelling + Industry Pricing:** industry pricing and voluntary disclosure are evaluated before the welfare comparison.
"""
)

# ---------------------------------------------------------------------------
# 4. Historical validation
# ---------------------------------------------------------------------------
st.header("4. Empirical Comparison: Observed vs. Scenario Equilibria")
observed = observed_market_outcome(df, consumers, tech, s=s)
c = next((r for r in results if r.get("Code") == "C" and "regular_price" in r), None)
ci = next((r for r in results if r.get("Code") == "CI" and "regular_price" in r), None)

summary = df[["T", "p_regular", "p_premium"]].rename(
    columns={"T": "Week", "p_regular": "Observed p₀", "p_premium": "Observed p₁"}
).copy()

if c and ci:
    summary["C p₀"] = c["regular_price"]
    summary["C p₁"] = c["premium_price"]
    summary["CI p₀"] = ci["regular_price"]
    summary["CI p₁"] = ci["premium_price"]
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
            ["Scenario C", c["p₀"], c["p₁"], c["Industry profit"], c["Welfare"]],
            ["Scenario CI", ci["p₀"], ci["p₁"], ci["Industry profit"], ci["Welfare"]],
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
        dist = float(np.hypot(r["regular_price"] - mean_p0, r["premium_price"] - mean_p1))
        realism_rows.append(
            {
                "Scenario": r["Code"],
                "Equilibrium Regular p₀": r["p₀"],
                "Equilibrium Premium p₁": r["p₁"],
                "Distance from observed mean": dist,
                "Industry profit Π": r["Industry profit"],
                "Social welfare W": r["Welfare"],
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
