# Annotations

## Run the annotator

From `annotator/`, install dependencies once with `npm install`, then start the app with `npm run dev`. Open <http://localhost:3000>, enter your annotator ID, and use the same ID whenever you return.

## How to annotate

Annotate each message in the context of its conversation. Select one label; every click is saved automatically. When a conversation is complete, select **Next conversation**.

- `english`: predominantly English.
- `bangla`: Bangla written in Bengali script.
- `romanized_bangla`: Bangla written in Latin script.

The interface only offers labels compatible with the message's script. Bengali-script messages are pre-labelled as `bangla`; review and change them if needed.

## Annotators and agreement

Two humans independently annotated the corpus. Their files are stored by annotator ID in `data/annotations/annotator_1/` and `data/annotations/annotator_2/`.

- Annotator 1: 10,004 annotations across 1,500 conversations.
- Annotator 2: 10,004 annotations across 1,500 conversations.
- Paired message annotations: 10,004 (20,008 individual annotations total).
- Exact agreement: 9,658 / 10,004 (96.54%).
- Cohen's kappa (unweighted): 0.9412.
