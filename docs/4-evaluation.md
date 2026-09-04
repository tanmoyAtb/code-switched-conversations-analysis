# Language Identification Evaluation

This document compares three message-level language-identification approaches on the
resolved Bangla–English conversation dataset:

1. Character TF-IDF n-gram logistic regression.
2. Fine-tuned XLM-R base.
3. Zero-shot GPT-4o mini through the API.

The complete machine-readable records and prediction artifacts are in
[`training/results/`](../training/results/). Prediction records contain only
`message_id`, `true_label`, and `predicted_label`; they do not duplicate message text.

## Evaluation design

The fixed, conversation-disjoint splits contain 1,500 conversations and 10,004 labelled
messages:

| Split | Conversations | Messages | Purpose |
| --- | ---: | ---: | --- |
| Training | 1,200 | 8,058 | Fit the local models |
| Validation | 150 | 971 | Select configuration, epoch, or prompt |
| Test | 150 | 975 | One final held-out evaluation |

The labels are `english`, `bangla`, and `romanized_bangla`. Accuracy and macro-F1 are
reported on the test set. Macro-F1 gives equal weight to each language label despite the
test split having different label frequencies.

No test result was used to select a configuration, epoch, or prompt.

## Model selection

| Approach | Validation-only selection |
| --- | --- |
| n-gram | Selected `char_wb_2_5_c1`: character-within-word TF-IDF 2–5 grams, `C=1.0` (validation macro-F1: 98.99%). |
| XLM-R base | Fine-tuned for up to five epochs; selected epoch 5 (validation macro-F1: 99.33%). The selected model was then refit on training + validation. |
| GPT-4o mini | Compared two fixed zero-shot prompts. `decision_rules_v1` won with 86.83% validation macro-F1, ahead of `definitions_v1` at 82.29%. |

## Held-out test results

The test set has 207 English, 229 Bangla, and 539 romanized-Bangla messages.

| Model | Accuracy | Macro-F1 | English F1 | Bangla F1 | Romanized Bangla F1 | Errors / 975 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Character n-gram logistic regression | 98.56% | 98.43% | 96.57% | 100.00% | 98.71% | 14 |
| XLM-R base | **99.18%** | **99.10%** | **98.05%** | **100.00%** | **99.26%** | **8** |
| GPT-4o mini, zero-shot | 89.03% | 88.98% | 89.44% | 88.25% | 89.25% | 107 |

XLM-R base is the best-performing model in this evaluation. It exceeds the n-gram
baseline by 0.62 percentage points in accuracy and GPT-4o mini by 10.15 percentage
points.

## Error pattern

The principal error is distinguishing romanized Bangla from the other two labels. The
following matrices use rows as the true label and columns as the predicted label.

### Character n-gram logistic regression

| True \ Predicted | English | Bangla | Romanized Bangla |
| --- | ---: | ---: | ---: |
| English | 197 | 0 | 10 |
| Bangla | 0 | 229 | 0 |
| Romanized Bangla | 4 | 0 | 535 |

### XLM-R base

| True \ Predicted | English | Bangla | Romanized Bangla |
| --- | ---: | ---: | ---: |
| English | 201 | 0 | 6 |
| Bangla | 0 | 229 | 0 |
| Romanized Bangla | 2 | 0 | 537 |

### GPT-4o mini

| True \ Predicted | English | Bangla | Romanized Bangla |
| --- | ---: | ---: | ---: |
| English | 199 | 1 | 7 |
| Bangla | 0 | 229 | 0 |
| Romanized Bangla | 39 | 60 | 440 |

GPT-4o mini recognised every Bangla-script test message, but frequently assigned
romanized Bangla to English or Bangla. The explicit decision-rule prompt improved that
distinction over the simpler prompt, but did not close the gap with the models trained on
the annotated corpus.

## Interpretation

For this dataset and these three labels, the fine-tuned XLM-R model is the recommended
automatic classifier. The character n-gram baseline is also very strong, inexpensive to
rerun, and a useful production baseline. GPT-4o mini is a useful zero-shot reference, but
its lower accuracy makes it unsuitable as the primary labeler for this task without
additional adaptation or a different evaluation design.

## Reproducibility and limitations

- The n-gram and XLM-R results depend on the resolved annotations and fixed splits in
  [`data/resolved/`](../data/resolved/). The local models use random seed 42.
- XLM-R uses `FacebookAI/xlm-roberta-base`, a maximum input length of 96 tokens, and the
  training configuration recorded in its evaluation JSON. Its trained weights are kept
  locally in `training/models/xlm_roberta_base/` and are intentionally not committed.
- GPT-4o mini is evaluated zero-shot with the `gpt-4o-mini` alias. The exact prompt text
  and validation results are retained, but a future API rerun may differ if that alias is
  updated. The current run record is preserved in its evaluation JSON.
- These are point estimates from one held-out split of 975 messages. They do not establish
  performance on other datasets, domains, dialects, or annotation schemes, and no
  confidence intervals were calculated.

## Result artifacts

| Model | Evaluation record | Prediction artifacts |
| --- | --- | --- |
| n-gram | [`evaluation.json`](../training/results/ngram_logistic_regression/evaluation.json) | [`predictions/`](../training/results/ngram_logistic_regression/predictions/) |
| XLM-R base | [`evaluation.json`](../training/results/xlm_roberta_base/evaluation.json) | [`predictions/`](../training/results/xlm_roberta_base/predictions/) |
| GPT-4o mini | [`evaluation.json`](../training/results/gpt_4o_mini/evaluation.json) | [`predictions/`](../training/results/gpt_4o_mini/predictions/) |
