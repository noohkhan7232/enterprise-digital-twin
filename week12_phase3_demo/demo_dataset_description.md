# Demonstration Dataset Description

This document describes the example data used to drive the demonstration. The data is illustrative
and synthetic, chosen to exercise the platform's contracts and engines deterministically. It does
not represent any real organisation, and no benchmark or performance numbers are derived from it.

---

## 1. Purpose

The dataset exists to make the demonstration concrete and reproducible. Its goals are to (a) give the
digital twin, predictive, knowledge and workflow layers realistic shapes to operate on, and (b) feed
the production-engineering subsystems enough signal to show monitoring, reliability, SLO evaluation,
capacity forecasting and readiness assessment. Because the platform is deterministic, the same inputs
always produce the same outputs, so the demonstration is repeatable.

## 2. Example Enterprise Assets

A small fleet of industrial assets, each represented as a digital twin with a static model and a
dynamic state stream. Representative asset classes:

| Asset class | Example attributes (static) | State signals (dynamic) |
|-------------|-----------------------------|-------------------------|
| Rotating machinery (pump) | Identifier, model, install date, rated load | Vibration, temperature, run hours |
| Conveyor line segment | Identifier, length, motor spec | Throughput, motor current, stoppages |
| Process vessel | Identifier, capacity, material | Pressure, level, temperature |
| Edge gateway | Identifier, firmware, location | Connectivity, message rate, queue depth |

Assets carry stable identifiers so that telemetry, predictions and knowledge documents can all be
associated with the same physical thing.

## 3. Telemetry

Telemetry is modelled as time-ordered observations per asset signal. For the demonstration, signal
streams are generated deterministically (e.g., a steady baseline with controlled trends and injected
anomalies) so that:

- The **metrics engine** has series to aggregate, window and trend.
- The **monitoring** layer has distributions to compare against a baseline, including a deliberately
  introduced shift to show data-drift detection, and a changed input-to-output relationship to show
  concept-drift detection.
- The **predictive** layer has histories to forecast over.
- The **capacity planner** has resource-usage histories (e.g., request volume, storage growth) to
  project, including one series that approaches a configured limit to show exhaustion detection.

No claim is made that these streams reflect real-world distributions; they are constructed to
exercise the code paths clearly and repeatably.

## 4. Business Events

Discrete events accompany the telemetry to drive the workflow and incident layers:

| Event type | Role in the demo |
|------------|------------------|
| Work order created | Triggers a workflow spanning twin query, prediction and document retrieval |
| Maintenance completed | Resets an asset's health expectation; recorded in lineage |
| Threshold breach | Raises a monitoring alert and may open an incident |
| Release proposed | Exercises CI/CD quality gates and release validation |
| Deployment performed | Exercises deployment readiness and health verification |

Events are timestamped through the platform's injected clock, keeping the sequence deterministic.

## 5. Knowledge Corpus

A small, illustrative set of documents (e.g., maintenance procedures, equipment manuals, safety
notes) indexed for the retrieval-augmented knowledge layer. Documents are associated with asset
identifiers so retrieval can be scoped to the equipment a query concerns. The index is treated as a
versioned artefact so the evidence available at demonstration time is reproducible.

## 6. Expected Workflows

The demonstration exercises a representative end-to-end flow:

1. A **threshold breach** event arrives for a pump showing rising vibration and temperature.
2. The **monitoring** layer flags the change and raises an alert; an **incident** is opened.
3. A **workflow** is initiated: it queries the **digital twin** for current state, requests a
   **prediction** of remaining useful life, and **retrieves** the relevant maintenance procedure
   from the knowledge corpus.
4. An **agent** drafts a work order grounded in the retrieved procedure; the action is typed and
   recorded for audit.
5. The **MLOps** lineage records which model version produced the prediction and from which data.
6. The **observability** layer reflects the incident in reliability and SLO views, and the
   **incident manager** moves the incident through its lifecycle to resolution and postmortem.
7. The **production-readiness** assessment is run to confirm the platform remains fit to operate.

## 7. What This Dataset Does Not Provide

- It does not provide benchmark accuracy, latency or throughput figures, and none are claimed.
- It does not model a specific real organisation or proprietary data.
- It is sized for clarity and determinism in a live setting, not for scale testing.

For scale or accuracy evaluation, real datasets and sustained workloads would be required; this is
identified as future work in the research paper.