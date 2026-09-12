# Code-Switched Conversations Analysis

_Bangla–English business conversation analysis in collaboration with Engaze._

## Why this work

Customers communicate with businesses in more than one practical language mode: English,
Bangla written in Bengali script, and Bangla written in the Latin alphabet. An automated
reply that does not follow the customer's current language mode can make the conversation
less natural and may contribute to the customer changing how they communicate.

This project investigates that relationship and evaluates a practical solution: identify the
customer's language mode before generating an automated reply, then preserve that mode in
the response pipeline.

## Data

The analysis uses **10,004 anonymized messages** from **1,500 conversations** across
**30 businesses**. Each record includes the message order within its conversation, whether
the sender is a customer or business, whether a business reply was automated, and a resolved
human language annotation:

- `english` — 2,083 messages (20.82%)
- `bangla` — Bangla written in Bengali script; 2,207 messages (22.06%)
- `romanized_bangla` — Bangla written in Latin script; 5,714 messages (57.12%)

Romanized Bangla is the majority mode: 77.94% of messages are written entirely in Latin
script, and roughly three quarters of those are Bangla rather than English.

## What we found

We analysed 3,362 direct interaction sequences:

```text
customer message → business reply → next customer message
```

| Reply source | Reply matches the customer's preceding language | Customer switches language afterward |
| --- | ---: | ---: |
| Automated business reply (1,431 sequences) | 64.50% | 15.79% |
| Human-written business reply (1,931 sequences) | 73.49% | 11.81% |

Automated replies were less likely to match the customer's preceding language and were
followed by more customer language switching. The differences are statistically strong in
this corpus: `p = 2.39e-08` for reply-language alignment and `p = 0.00090` for the overall
customer-switch rate.

The clearer pattern is **language mismatch**, regardless of who sent the reply:

| Reply source | Customer switches after a matched reply | Customer switches after a mismatched reply |
| --- | ---: | ---: |
| Automated | 11.81% | 23.03% |
| Human-written | 8.46% | 21.09% |

When customers did switch after a mismatched reply, **89–91%** switched toward the business
reply's language. This supports the interpretation that reply-language alignment matters to
the customer's next language choice.

This is an observational finding, not proof that automation or a language model caused a
switch. Reply templates, business policy, conversation topic, and other unobserved factors
may also affect the relationship.

## Evaluating the language-identification component

We evaluated three language-identification approaches. Configuration, epoch, and prompt
selection used validation data; the final comparison used one held-out test set of 975
messages.

| Model | Test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| XLM-R base | **99.18%** | **99.10%** |
| Character n-gram logistic regression | 98.56% | 98.43% |
| GPT-4o mini, zero-shot | 89.03% | 88.98% |

This evaluates the language-identification component, not the effect of deploying it in a
live conversation. An A/B deployment study would be needed to establish whether using the
classifier reduces customer language switching.

## Conclusion

Language alignment is a concrete opportunity to improve automated business conversations.
The evidence indicates that customers are more likely to change language after a business
reply that does not match their preceding language mode, while automated replies are less
aligned overall.

The recommended solution is to run a language classifier before automated response
generation and pass the detected mode—English, Bangla script, or romanized Bangla—into the
reply pipeline. XLM-R base is the preferred classifier for accuracy; the character n-gram
model is a strong lightweight alternative.

## Documentation

1. [Corpus profile](./docs/1-corpus-profile.md)
2. [Code-switching tendency analysis](./docs/2-code-switch-tendency.md)
3. [Annotations](./docs/3-annotations.md)
4. [Language-identification evaluation](./docs/4-evaluation.md)
