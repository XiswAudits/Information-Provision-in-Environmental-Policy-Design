# Information Provision in Environmental Policy Design — Empirical Coffee Panel

A clean Streamlit implementation of the two-tier (Regular/Premium) case of **Danilina & Grigoriev (2020), _Information Provision in Environmental Policy Design_**, operationalized on the supplied 11-week purchase panel of three households.

## What is implemented

1. **Data extraction** — weekly Regular/Premium prices and three household purchase quantities are stored in `data.csv`.
2. **Consumer calibration** — positive-demand quantities are averaged by household/type; budgets are anchored at maximum observed expenditure; the identifiable premium-vs-regular myopia difference is estimated with a small regularized switching likelihood.
3. **Price geometry** — all box and indifference hyperplanes are generated and their pairwise 2-D intersections are enumerated.
4. **Four scenarios** — C, L, CI and LI are solved by finite enumeration of price vertices, following the paper's K=1 scenario logic.
5. **Validation** — observed weekly prices are compared with Scenario C and CI equilibrium prices; model-implied demand, profit, welfare and footprint are reported.

## Empirical identification caveat

The paper assumes consumer budgets, myopia, production costs and environmental footprints are known. The supplied purchase panel only identifies prices, quantities and switching. Therefore this repository **does not fabricate environmental data**:

- budgets are estimated from observed expenditure bounds;
- only the identifiable μ₁−μ₀ difference is estimated from switching, with μ₀ normalized to zero;
- variable costs are calibrated as a configurable share of observed price floors;
- footprints are explicit UI inputs because the dataset contains no environmental-impact measurements.

The app reports switching-fit diagnostics so the deterministic model's empirical limitations remain visible.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reference

Danilina, V. & Grigoriev, A. (2020). *Information Provision in Environmental Policy Design*. Journal of Environmental Informatics, 36(1), 1–10. DOI: 10.3808/jei.201900410.
