# Annotations

## Run the annotator

From `annotator/`, install dependencies once with `npm install`, then start the app with `npm run dev`. Open <http://localhost:3000>, enter your annotator ID, and use the same ID whenever you return.

## How to annotate

Annotate each message in the context of its conversation. Select one label; every click is saved automatically. When a conversation is complete, select **Next conversation**.

- `english`: predominantly English.
- `bangla`: Bangla written in Bengali script.
- `romanized_bangla`: Bangla written in Latin script.

The interface only offers labels compatible with the message's script. Bengali-script messages are pre-labelled as `bangla`; review and change them if needed.

In the final corpus, all 2,151 Bengali-only messages and all 56 messages containing both
scripts are resolved as `bangla`. The substantive language-mode distinction is therefore
between `english` and `romanized_bangla` for Latin-script messages.

## Annotators and agreement

Two humans independently annotated the corpus. Their files are stored by annotator ID in `data/annotations/annotator_1/` and `data/annotations/annotator_2/`.

- Annotator 1: 10,004 annotations across 1,500 conversations.
- Annotator 2: 10,004 annotations across 1,500 conversations.
- Paired message annotations: 10,004 (20,008 individual annotations total).
- Exact agreement: 9,658 / 10,004 (96.54%).
- Cohen's kappa (unweighted): 0.9412.

The 346 disagreements were discussed and settled separately by the two annotators before
the resolved labels were finalized. On the 7,797 Latin-only messages, exact agreement was
95.56% and Cohen's kappa was 0.8898.

## Resolved label distribution

The 10,004 resolved labels are unbalanced, with romanized Bangla the majority mode:

| Label | Messages | Share |
| --- | ---: | ---: |
| `romanized_bangla` | 5,714 | 57.12% |
| `bangla` | 2,207 | 22.06% |
| `english` | 2,083 | 20.82% |

The `bangla` total is the 2,151 Bengali-only messages plus the 56 mixed-script messages
resolved as `bangla`. The remaining 7,797 Latin-only messages divide into 5,714 romanized
Bangla (73.29%) and 2,083 English (26.71%).

Customers and businesses use the three modes in broadly similar proportions:

| Side | `english` | `bangla` | `romanized_bangla` |
| --- | ---: | ---: | ---: |
| Customer (5,415) | 20.28% | 20.79% | 58.93% |
| Business (4,589) | 21.46% | 23.56% | 54.98% |
