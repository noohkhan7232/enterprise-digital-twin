# Week 5 Figures — Predictive Maintenance Intelligence Subsystem

Publication-quality figures for the Week 5 research report of the **Enterprise
Digital Twin & Predictive Maintenance Intelligence Platform**. All figures are
rendered at 300 DPI on a white background, export-ready for the report PDF and
the GitHub README. Every quantitative figure is computed from the actual
parameters of the delivered `src/predictive/` modules (cost model 5,000 / 50,000;
break-even P = 0.10; score bands 2 / 5 / 8 / 12; risk thresholds 0.25 / 0.50 / 0.75;
confidence weights 0.40 / 0.35 / 0.25).

| # | File | Title | Type |
|--:|------|-------|------|
| 1 | `fig1_pipeline.png` | End-to-End Predictive Maintenance Pipeline | Flow diagram |
| 2 | `fig2_decision_architecture.png` | Maintenance Decision Architecture | Block diagram |
| 3 | `fig3_rul_vs_failure_probability.png` | RUL vs Failure Probability Relationship | Analytical plot |
| 4 | `fig4_decision_matrix.png` | Maintenance Action Decision Matrix | Heatmap |
| 5 | `fig5_cost_analysis.png` | Maintenance Cost vs Failure Cost Analysis | Bar + crossover |
| 6 | `fig6_expected_savings.png` | Expected Savings vs Failure Probability | Analytical plot |
| 7 | `fig7_confidence_components.png` | Decision Confidence Components | Donut + curve |
| 8 | `fig8_system_architecture.png` | Predictive Maintenance System Architecture | Layered diagram |
| 9 | `fig9_priority_escalation.png` | Maintenance Priority Escalation Flowchart | Flowchart |
| 10 | `fig10_dashboard_mockup.png` | Enterprise Predictive Maintenance Dashboard | UI mockup |

## Figure notes

- **Figure 1** traces the five registered engines (`acoustic_health_index` →
  `acoustic_health_trend` → `trajectory_rul_predictor` → `survival_failure_risk`
  → `acoustic_maintenance_agent`) and annotates the `should_predict_rul()` gate.
- **Figure 3** overlays the exponential (constant-hazard) and Weibull (wear-out)
  survival models at a 30-day horizon, with the four risk bands shaded.
- **Figure 4** renders the action surface over the (health, failure-probability)
  plane, computed directly from the delivered rule engine under a degrading
  trend (hence NO_ACTION, reserved for non-degrading machines, does not appear).
- **Figures 5 and 6** make the cost-benefit model explicit: the expected-cost
  crossover and the linear expected-savings curve, both crossing at the
  break-even probability P* = maintenance_cost / failure_cost = 0.10.
- **Figure 7** shows the confidence blend (0.40·trend + 0.35·RUL +
  0.25·agreement) and how each component responds to RUL interval width and
  signal agreement.
- **Figure 9** encodes the exact severity bands (2 / 5 / 8 / 12) and the
  risk/warning-driven priority floor.
- **Figure 10** is a representative operator console: KPI cards, health
  extrapolation with confidence band, horizon risk, a recommendation banner with
  cost/savings/downtime, and a rule-level explainability panel.

## Palette

A consistent enterprise palette is used across all figures: deep navy
(`#1A2B47`) for structure, steel/teal for the pipeline, and a green→amber→red
severity ramp for status and actions.
