# Evaluation Report

Generated 2026-09-14 14:13 against 35 goldens.

Metrics are split by what they read. Retrieval metrics look only at
`retrieval_context`; generation metrics look only at the answer. That split is
the point: it makes a failure attributable to fetching the wrong facts or to
writing a bad answer over the right ones, which need different fixes.

## Golden set

| | |
|---|---|
| Total | 35 |
| By category | both: 7, graph_multihop: 15, semantic: 8, unanswerable: 5 |
| Multi-hop (2+) | 13 |
| Must abstain | 5 |

## Retrieval quality

| Metric | n | Mean | Median | Min | Threshold | Pass rate |
|---|---:|---:|---:|---:|---:|---:|
| ContextualPrecision | 30 | 0.73 | 0.88 | 0.00 | >= 0.65 | 67% |
| ContextualRecall | 30 | 0.92 | 1.00 | 0.00 | >= 0.80 | 87% |
| ContextualRelevancy | 30 | 0.43 | 0.39 | 0.00 | >= 0.35 | 53% |
| GraphPathRecall | 35 | 1.00 | 1.00 | 1.00 | >= 1.00 | 100% |
| RoutingAccuracy | 35 | 1.00 | 1.00 | 1.00 | >= 1.00 | 100% |

## Generation quality

| Metric | n | Mean | Median | Min | Threshold | Pass rate |
|---|---:|---:|---:|---:|---:|---:|
| AgronomicUsefulness | 30 | 0.93 | 0.90 | 0.70 | >= 0.75 | 97% |
| AnswerRelevancy | 30 | 0.91 | 1.00 | 0.00 | >= 0.75 | 90% |
| CorrectAbstention | 35 | 1.00 | 1.00 | 1.00 | >= 1.00 | 100% |
| Faithfulness | 30 | 1.00 | 1.00 | 0.90 | >= 0.90 | 100% |
| Hallucination | 30 | 0.90 | 1.00 | 0.00 | >= 0.75 | 87% |

## Reading these numbers

- **RoutingAccuracy, GraphPathRecall, CorrectAbstention** are deterministic and
  free. They gate every CI run, and when a judged score moves they are how you
  tell a real regression from judge variance.
- **ContextualRelevancy scores lowest by design.** A graph traversal returns
  every row matching the pattern, and rows that are correct but not needed for
  this particular question count against it. That is a real cost of graph
  retrieval; the threshold reflects it rather than hiding it.
- **Hallucination is scored against hand-written ground truth**, not against
  what retrieval returned. An answer can be perfectly faithful to bad context
  and still be false, and only this metric separates the two. Lower is better.
- **Abstention cases carry no contextual scores.** They have no retrieval
  context by design, so the contextual metrics have nothing to measure.

## Below threshold

| Golden | Metric | Score | Reason |
|---|---|---:|---|
| `bt_limitations` | AnswerRelevancy | 0.00 | metric error: judge could not produce a valid AnswerRelevancyScoreReason in 3 attempts |
| `bt_limitations` | ContextualPrecision | 0.33 | The score is 0.33 because the only relevant node is ranked third rather than first, which lowers precision. The first node in the retrieval contexts is irreleva |
| `bt_limitations` | ContextualRelevancy | 0.00 | metric error: judge could not produce a valid ContextualRelevancyVerdicts in 3 attempts |
| `chickpea_ascochyta` | ContextualRecall | 0.67 | The score is 0.67 because sentence 1 (susceptibility to ascochyta blight at flowering and cotton bollworm at podding) is fully supported: node 3 in retrieval co |
| `chickpea_ascochyta` | ContextualRelevancy | 0.33 | The score is 0.33 because while some retrieved statements directly address the input—e.g., "Ascochyta blight (Ascochyta rabiei, fungal_disease) attacks Chickpea |
| `chickpea_nitrogen_credit` | ContextualPrecision | 0.63 | The score is 0.63 because the relevant nodes are not consistently ranked at the top, which lowers precision despite good early placement. The first node is rele |
| `chickpea_nitrogen_credit` | ContextualRelevancy | 0.27 | The score is 0.27 because while some retrieved statements directly address the input—e.g., "A chickpea crop ahead of wheat or maize typically justifies reducing |
| `cotton_pests_cukurova` | ContextualRelevancy | 0.17 | The score is 0.17 because most retrieved statements miss the mark: several describe mitigation tactics instead of pests (e.g., 'Mitigated by IPM scouting and th |
| `cover_cropping_sandy_semiarid` | ContextualPrecision | 0.59 | The score is 0.59 because the relevant nodes are not consistently ranked above the irrelevant ones. The first node in retrieval contexts is irrelevant, only exp |
| `cover_cropping_sandy_semiarid` | ContextualRelevancy | 0.33 | The score is 0.33 because although a few statements directly explain the benefit—e.g. 'Living roots hold aggregates together and keep the surface rough enough t |
| `drip_irrigation_cotton_why` | AgronomicUsefulness | 0.70 | The response correctly names cotton-specific benefits and mirrors the retrieval context closely: dry canopy reducing foliar fungal disease infection periods, in |
| `drip_irrigation_cotton_why` | ContextualPrecision | 0.28 | The score is 0.28 because the relevant nodes are concentrated toward the bottom of the ranking rather than the top, which hurts precision. The first two nodes i |
| `drip_irrigation_cotton_why` | Hallucination | 0.60 | The score is 0.60 because the actual output correctly aligns with the context on three points—dry canopy reducing foliar fungal disease risk, dry inter-row supp |
| `fungicide_resistance` | Hallucination | 0.00 | metric error: judge could not produce a valid Verdicts in 3 attempts |
| `humid_subtropical_crops` | ContextualPrecision | 0.08 | The score is 0.08 because the only relevant node, ranked 13th in the retrieval contexts, is buried beneath twelve irrelevant nodes that all precede it. For inst |
| `humid_subtropical_crops` | ContextualRelevancy | 0.05 | The score is 0.05 because almost all retrieved statements describe crops grown in unrelated climates such as 'Hot semi-arid climate' (e.g., barley and chickpea  |
| `ipm_principles` | ContextualPrecision | 0.00 | metric error: judge could not produce a valid ContextualPrecisionScoreReason in 3 attempts |
| `ipm_principles` | ContextualRelevancy | 0.00 | metric error: judge could not produce a valid ContextualRelevancyVerdicts in 3 attempts |
| `maize_konya_organic` | ContextualPrecision | 0.00 | metric error: judge could not produce a valid ContextualPrecisionScoreReason in 3 attempts |
| `maize_konya_organic` | ContextualRelevancy | 0.28 | The score is 0.28 because although some highly relevant facts are present—e.g. "European corn borer is controlled by Bacillus thuringiensis (biocontrol, organic |
| `maize_konya_restricted` | ContextualRecall | 0.50 | The score is 0.50 because the first part of the expected output, stating that Chlorpyrifos-ethyl, deltamethrin and lambda-cyhalothrin are governed by the Turkis |
| `maize_konya_restricted` | ContextualRelevancy | 0.00 | metric error: judge could not produce a valid ContextualRelevancyScoreReason in 3 attempts |
| `olive_fruit_fly_treatments` | AnswerRelevancy | 0.60 | The score is 0.60 because while the output does identify some products/methods that control the olive fruit fly and notes their organic status (directly answeri |
| `olive_fruit_fly_treatments` | ContextualPrecision | 0.50 | The score is 0.50 because the first node in retrieval contexts is irrelevant, being ranked higher than the relevant one. Its reason explains that it "discusses  |
| `olive_fruit_fly_treatments` | ContextualRecall | 0.50 | The score is 0.50 because sentence 1 (Spinosad details) is fully supported by node 2 in retrieval context, which states the exact same efficacy, application rat |
| `olive_fruit_fly_treatments` | Hallucination | 0.50 | The score is 0.50 because while the actual output correctly aligns with the context regarding spinosad's application rate, efficacy, pre-harvest interval, and o |
| `pheromone_trapping_olive_why` | ContextualRelevancy | 0.32 | The score is 0.32 because although some retrieved statements directly answer the input—e.g., 'The olive fruit fly meets all three conditions (single dominant ho |
| `rotation_verticillium_slow` | ContextualRelevancy | 0.33 | The score is 0.33 because although the context contains highly relevant statements explaining the core mechanism—'Verticillium dahliae forms microsclerotia that |
| `sugarbeet_cercospora` | Hallucination | 0.67 | The score is 0.67 because the actual output correctly aligns with the context on two factual points—azoxystrobin (0.8 L/ha, high efficacy, 28-day PHI) and tebuc |
| `sunn_pest_quality_damage` | ContextualRelevancy | 0.21 | The score is 0.21 because most retrieved statements are off-topic, covering unrelated pests and diseases like 'yellow rust', a chickpea disease, maize pests, an |
| `verticillium_practices` | AnswerRelevancy | 0.00 | metric error: judge could not produce a valid Verdicts in 3 attempts |
| `wheat_clay_practices_rust` | ContextualRelevancy | 0.17 | The score is 0.17 because the vast majority of the retrieval context is off-topic, covering unrelated crops, pests, and regions (e.g., "fall armyworm migration, |
| `wheat_rotation` | ContextualRelevancy | 0.33 | The score is 0.33 because although the context contains directly relevant statements such as 'Wheat rotates well with Chickpea (legume, spring): The legume fixe |
| `whitefly_options` | ContextualPrecision | 0.33 | The score is 0.33 because the irrelevant nodes are ranked above the relevant ones. The first node in retrieval contexts discusses 'Cotton bollworm and Bacillus  |
| `yellow_rust_control` | ContextualPrecision | 0.27 | The score is 0.27 because the relevant nodes were ranked lower than several irrelevant ones. The first node in retrieval contexts discusses Septoria leaf blotch |
| `yellow_rust_control` | ContextualRecall | 0.00 | The score is 0.00 because the single sentence in the expected output cannot be fully attributed to the nodes in retrieval context. While node 6 supports the cla |

## Reproducing

```bash
docker compose up -d
uv run agrirag-seed --reset && uv run agrirag-index --extract
uv run python -m evals.runner --refresh   # run the agent over the goldens
uv run python -m evals.report             # score and regenerate this file
```
