from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


@dataclass
class ConsumerParams:
    name: str
    d0: float
    d1: float
    b0: float
    b1: float
    mu0: float
    mu1: float
    fit_accuracy: float


@dataclass
class TechParams:
    c0: float
    c1: float
    phi0: float
    phi1: float
    gamma: float


def load_data(path: str = "data.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def _positive_mean(x: np.ndarray, fallback: float = 1.0) -> float:
    values = x[x > 0]
    return float(values.mean()) if len(values) else fallback


def _estimate_mu_delta(q0, q1, p0, p1, d0, d1, b0, b1, s=2.0,
                       tau=10.0, reg=0.05):
    """Estimate mu1-mu0 from observed premium-vs-regular switching.

    The paper's deterministic utility is the structural target. Because this
    11-week panel contains noisy/inconsistent switches, a small regularized
    logistic likelihood is used only for calibration. The resulting parameters
    are then evaluated with the paper's deterministic equilibrium rules.
    """
    y = (q1 > 0).astype(float)
    relative_cost = p1 * d1 - p0 * d0

    def loss(delta):
        z = ((b1 - b0) + delta * s - relative_cost) / tau
        return float(
            np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
            + reg * delta**2
        )

    result = minimize_scalar(loss, bounds=(-100, 100), method="bounded")
    delta = float(result.x)

    utility_diff = (b1 - b0) + delta * s - relative_cost
    predicted = utility_diff >= 0
    active = (q0 + q1) > 0
    actual = q1 > 0
    accuracy = float((predicted[active] == actual[active]).mean()) if active.any() else 0.0
    return delta, accuracy


def estimate_consumers(df: pd.DataFrame, s: float = 2.0) -> List[ConsumerParams]:
    p0 = df.p_regular.to_numpy(float)
    p1 = df.p_premium.to_numpy(float)
    consumers = []

    for idx in range(1, 4):
        q0 = df[f"h{idx}_regular"].to_numpy(float)
        q1 = df[f"h{idx}_premium"].to_numpy(float)

        d0 = _positive_mean(q0, 1.0)
        d1 = _positive_mean(q1, 1.0)

        # Observed expenditure is a lower bound on latent WTP budget.
        # We use the maximum observed expenditure as the transparent
        # nonparametric budget estimate.
        b0 = float(np.max(p0 * q0))
        b1 = float(np.max(p1 * q1))

        delta, accuracy = _estimate_mu_delta(
            q0, q1, p0, p1, d0, d1, b0, b1, s=s
        )

        # Only mu1-mu0 is identified from binary switching. Normalize mu0=0.
        consumers.append(
            ConsumerParams(
                name=f"Household {idx}",
                d0=d0, d1=d1, b0=b0, b1=b1,
                mu0=0.0, mu1=delta, fit_accuracy=accuracy
            )
        )
    return consumers


def price_vertices(consumers: List[ConsumerParams], s: float,
                   include_zero: bool = True) -> pd.DataFrame:
    """Enumerate all 2-D intersections of box and indifference hyperplanes."""
    lines = []

    for c in consumers:
        # Box: p0*d0 = b0 + mu0*s
        lines.append(("box", c.name, c.d0, 0.0, c.b0 + c.mu0 * s))
        # Box: p1*d1 = b1 + mu1*s
        lines.append(("box", c.name, 0.0, c.d1, c.b1 + c.mu1 * s))
        # Indifference: p0*d0 - p1*d1 =
        # (b0-b1) + (mu0-mu1)*s
        lines.append((
            "indiff", c.name, c.d0, -c.d1,
            (c.b0 - c.b1) + (c.mu0 - c.mu1) * s
        ))

    points = []
    for a, b in combinations(lines, 2):
        A = np.array([[a[2], a[3]], [b[2], b[3]]], dtype=float)
        rhs = np.array([a[4], b[4]], dtype=float)
        if abs(np.linalg.det(A)) < 1e-9:
            continue

        p = np.linalg.solve(A, rhs)
        if np.all(np.isfinite(p)) and (include_zero or np.all(p >= 0)):
            points.append({
                "p_regular": float(p[0]),
                "p_premium": float(p[1]),
                "line_1": f"{a[0]}:{a[1]}",
                "line_2": f"{b[0]}:{b[1]}",
            })

    out = pd.DataFrame(points)
    if out.empty:
        return out
    return (
        out.drop_duplicates(["p_regular", "p_premium"])
        .sort_values(["p_regular", "p_premium"])
        .reset_index(drop=True)
    )


def market_outcome(p0: float, p1: float, consumers: List[ConsumerParams],
                   s: float, tech: TechParams,
                   assignments: Tuple[int, int] = (0, 1)) -> Dict:
    """Evaluate demand, profit, footprint and welfare at a price vertex.

    assignments maps physical technology 0/1 to the market label 0/1.
    Voluntary labelling can therefore let the clean technology use label 0.
    """
    demand = {0: 0.0, 1: 0.0}
    choices, utilities = {}, {}
    consumer_utility = 0.0

    for c in consumers:
        utilities_by_tech = []

        for tech_idx, label in enumerate(assignments):
            price = p0 if label == 0 else p1
            d = c.d0 if tech_idx == 0 else c.d1
            b = c.b0 if tech_idx == 0 else c.b1
            mu = c.mu0 if label == 0 else c.mu1
            utilities_by_tech.append(b - price * d + mu * s)

        vals = [0.0] + utilities_by_tech
        chosen = int(np.argmax(vals))

        if chosen == 0:
            choices[c.name] = -1
            utilities[c.name] = 0.0
        else:
            tech_idx = chosen - 1
            choices[c.name] = tech_idx
            demand[tech_idx] += c.d0 if tech_idx == 0 else c.d1
            utilities[c.name] = float(vals[chosen])

        consumer_utility += log(utilities[c.name] + 1.0)

    price0 = p0 if assignments[0] == 0 else p1
    price1 = p0 if assignments[1] == 0 else p1

    profit0 = (price0 - tech.c0) * demand[0]
    profit1 = (price1 - tech.c1) * demand[1]
    profit = profit0 + profit1

    footprint = tech.phi0 * demand[0] + tech.phi1 * demand[1]
    welfare = consumer_utility + profit - tech.gamma * footprint

    return {
        "p_regular": p0,
        "p_premium": p1,
        "demand_regular": demand[0],
        "demand_premium": demand[1],
        "profit": profit,
        "footprint": footprint,
        "consumer_utility": consumer_utility,
        "welfare": welfare,
        "choices": choices,
        "utilities": utilities,
        "assignments": assignments,
    }


def candidate_price_grid(consumers: List[ConsumerParams], s: float) -> pd.DataFrame:
    vertices = price_vertices(consumers, s, include_zero=False)
    if vertices.empty:
        return vertices
    return vertices[
        (vertices.p_regular > 0) & (vertices.p_premium > 0)
    ].copy()


def solve_scenario(scenario: str, consumers: List[ConsumerParams],
                   tech: TechParams, s: float = 2.0) -> Dict:
    """Finite K=1 implementation of Scenarios C, L, CI and LI.

    C: government controls prices and welfare is maximized.
    L: government controls prices; producers choose the more profitable
       feasible label assignment at those prices.
    CI: industry chooses the profit-maximizing price vertex.
    LI: producers choose a feasible label assignment first; industry then
        chooses the profit-maximizing price for that assignment.
    """
    vertices = candidate_price_grid(consumers, s)
    if vertices.empty:
        raise ValueError("No finite price vertices were generated.")

    mandatory = [(0, 1)]
    voluntary = [(0, 1), (0, 0)]
    best = None

    if scenario == "C":
        candidates = [
            market_outcome(r.p_regular, r.p_premium, consumers, s, tech, mandatory[0])
            for r in vertices.itertuples(index=False)
        ]
        best = max(candidates, key=lambda x: (x["welfare"], x["profit"]))

    elif scenario == "L":
        for r in vertices.itertuples(index=False):
            producer_options = [
                market_outcome(r.p_regular, r.p_premium, consumers, s, tech, a)
                for a in voluntary
            ]
            response = max(producer_options, key=lambda x: (x["profit"], x["welfare"]))
            if best is None or (response["welfare"], response["profit"]) > (
                best["welfare"], best["profit"]
            ):
                best = response

    elif scenario == "CI":
        candidates = [
            market_outcome(r.p_regular, r.p_premium, consumers, s, tech, mandatory[0])
            for r in vertices.itertuples(index=False)
        ]
        best = max(candidates, key=lambda x: (x["profit"], x["welfare"]))

    elif scenario == "LI":
        responses = []
        for assignment in voluntary:
            candidates = [
                market_outcome(r.p_regular, r.p_premium, consumers, s, tech, assignment)
                for r in vertices.itertuples(index=False)
            ]
            responses.append(max(candidates, key=lambda x: (x["profit"], x["welfare"])))
        best = max(responses, key=lambda x: (x["welfare"], x["profit"]))

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    result = dict(best)
    result["scenario"] = scenario
    result["delimiter"] = (
        None if tech.phi0 == tech.phi1
        else round((tech.phi0 + tech.phi1) / 2.0, 8)
    )
    result["stringency"] = s
    return result


def observed_market_outcome(df: pd.DataFrame,
                            consumers: List[ConsumerParams],
                            tech: TechParams, s: float = 2.0) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        out = market_outcome(
            row.p_regular, row.p_premium, consumers, s, tech, (0, 1)
        )
        rows.append({
            "week": int(row.T),
            **{k: v for k, v in out.items()
               if k in ["p_regular", "p_premium", "profit", "welfare", "footprint",
                        "demand_regular", "demand_premium"]}
        })
    return pd.DataFrame(rows)
