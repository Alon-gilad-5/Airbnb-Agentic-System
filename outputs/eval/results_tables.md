# Results Tables

## Table 1. Dataset profile

| Regions | Total Reviews | Unique Properties | Unique Reviewers | Duplicate Rows |
|---:|---:|---:|---:|---:|
| 8 | 3847500 | 62225 | 2939714 | 14 |

## Table 2. Per-agent quality and reliability metrics

| Agent | n_primary / n_total | Primary Metric (95% CI) | Task Success (x/n, 95% CI) | Crash-Free | Contract | Step Trace | P95 Latency (ms) | Cases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| analyst | 4/8 | 1.0000 (95% CI 0.510-1.000) | 1.0000 (8/8; 95% CI 0.676-1.000) | 1.0000 | 1.0000 | 1.0000 | 0.424 | 8 |
| mail | 9/9 | 1.0000 (95% CI 0.701-1.000) | 0.7778 (7/9; 95% CI 0.453-0.937) | 1.0000 | 1.0000 | 1.0000 | 1.401 | 9 |
| market_watch | 4/8 | 1.0000 (95% CI 0.510-1.000) | 1.0000 (8/8; 95% CI 0.676-1.000) | 1.0000 | 1.0000 | 1.0000 | 3.128 | 8 |
| pricing | 4/8 | 1.0000 (95% CI 0.510-1.000) | 0.8750 (7/8; 95% CI 0.529-0.978) | 1.0000 | 1.0000 | 1.0000 | 0.469 | 8 |
| reviews | 18/18 | 0.6667 (95% CI 0.437-0.837) | 0.6667 (12/18; 95% CI 0.437-0.837) | 1.0000 | 1.0000 | 1.0000 | 0.012 | 18 |

Note: `n_primary` counts only cases where the agent's primary metric is defined (typically non-error-path checks).

## Table 3. Reviews threshold ablation

| Setting | Threshold | Answer Decision Accuracy | Evidence Precision | Task Success |
|---|---:|---:|---:|---:|
| Baseline | 0.4 | 0.4444 | 0.0750 | 0.4444 |
| Tuned (dev-selected) | 0.48 | 0.6667 | 0.1951 | 0.6667 |

## Manual rubric scoring status

- Reviews rubric status: `scored` (scored_cases=6).
- Mail rubric status: `scored` (scored_cases=4).

## Limitations

- Manual rubric scoring is single-annotator in this cycle.
- Evaluation is offline-first with synthetic fixtures for analyst/pricing/market-watch.
- Reported results support task quality + reliability claims, not direct business-impact claims.
