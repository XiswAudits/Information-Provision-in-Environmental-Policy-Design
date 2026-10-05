# Information Provision in Environmental Policy Design — Empirical Coffee Panel

A clean Streamlit implementation of the two-tier (Regular/Premium) case of **Danilina & Grigoriev (2020), _Information Provision in Environmental Policy Design_**, operationalized on the supplied 11-week purchase panel of three households.

## What is implemented

1. **Data extraction** — weekly Regular/Premium prices and all three household purchase quantities are stored in `data.csv`.
2. **Consumer calibration** — positive-demand quantities are averaged by household/type; budgets are anchored at maximum observed expenditure; the identifiable premium-vs-regular myopia difference is estimated with a regularized switching likelihood.
3. **Price-space geometry** — the app constructs the bounding hyperplanes and indifference hyperplanes requested in the model, calculates their 2-D intersections, overlays the observed weekly prices, and reports the resulting empirical demand-region partition.
4. **Three-level structure** — Government → Industry/Producers → Consumers is explicitly represented in the UI and in the scenario solver.
5. **Four scenarios** — C, L, CI and LI are evaluated over the feasible price vertices for the K=1, M=2 empirical adaptation.
6. **Equilibrium metrics** — prices, market coverage, household choices, aggregate industry profit Π, environmental footprint Φ, stringency s and social welfare W are reported.
7. **Historical validation** — observed weekly prices are compared with Scenario C and Scenario CI equilibrium prices, including per-week Euclidean price distance and an observed-vs-predicted summary table.
8. **Empirical partition** — the app includes the requested high-price, intermediate and low-Premium-price regions, with observed weeks and household behavior.

## Empirical identification caveat

The paper's general model treats consumer budgets, myopia, production costs and environmental footprints as model inputs. The supplied purchase panel only identifies prices, quantities, expenditures and switching. Therefore this repository **does not fabricate environmental data**:

- budgets are estimated from observed expenditure bounds;
- only the identifiable μ₁−μ₀ difference is estimated from switching, with μ₀ normalized to zero;
- variable costs are calibrated as a configurable share of observed price floors;
- footprints and the environmental weight are explicit UI inputs because the dataset contains no direct footprint measurements.

The region thresholds shown in the UI (such as p₁≈75 and p₁≈81) are **empirical switch thresholds from the observed panel**, not claimed to be exact structural parameters of the paper.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reference

Danilina, V. & Grigoriev, A. (2020). *Information Provision in Environmental Policy Design*. Journal of Environmental Informatics, 36(1), 1–10. DOI: 10.3808/jei.201900410.
