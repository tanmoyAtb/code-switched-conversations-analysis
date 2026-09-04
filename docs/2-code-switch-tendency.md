# Code-Switching Tendency — Human and Automated Replies

_A descriptive analysis of resolved language annotations across anonymized Bangla–English
business conversations._

## 1. Scope and sequence definition

This analysis uses all 10,004 resolved human annotations across 1,500 conversations. It
does not use predictions from the n-gram, XLM-R, or GPT-4o mini models.

To give each business reply an unambiguous preceding customer language and subsequent
customer response, the analysis retains only immediately adjacent sequences of the form:

```text
customer message → business reply → next customer message
```

There are 3,362 such sequences: 1,431 with an automated business reply and 1,931 with a
human-written business reply.

For every sequence:

- **Reply-language match** means the business reply has the same language label as the
  preceding customer message.
- **Customer switch** means the next customer message has a different label from the
  preceding customer message.
- **Switch toward reply** means the next customer message has the business reply's label
  after a mismatched reply.

Labels are `english`, `bangla`, and `romanized_bangla`.

## 2. Headline result

| Reply source | Sequences | Reply matches preceding customer language | Customer switches language afterward |
| --- | ---: | ---: | ---: |
| Automated business reply | 1,431 | 923 (64.50%) | 226 (15.79%) |
| Human-written business reply | 1,931 | 1,419 (73.49%) | 228 (11.81%) |

Automated replies match the preceding customer language 8.99 percentage points less often
than human-written replies. Customers also switch language 3.98 percentage points more
often after automated replies. Two-sided Fisher exact tests give `p = 2.39e-08` for the
reply-alignment difference and `p = 0.00090` for the overall customer-switch difference.

## 3. Mismatch is the stronger pattern

For both reply sources, a reply-language mismatch is associated with a substantially higher
customer switch rate.

| Reply source | Matched reply: customer switch | Mismatched reply: customer switch |
| --- | ---: | ---: |
| Automated | 109 / 923 (11.81%) | 117 / 508 (23.03%) |
| Human-written | 120 / 1,419 (8.46%) | 108 / 512 (21.09%) |

The mismatch-versus-match association is strong for automated replies (`p = 5.82e-08`) and
for human-written replies (`p = 7.10e-13`).

Among mismatched replies, however, the customer-switch rates are similar: 23.03% after an
automated reply and 21.09% after a human-written reply (`p = 0.497`). This indicates that
the higher overall switching observed after automated replies is consistent primarily with
their lower rate of language alignment, rather than evidence that customers react
differently merely because a reply is automated.

## 4. Direction of switching

When customers change language after a mismatched reply, they usually move toward the
business reply's language.

| Reply source | Customer switches after mismatch | Switches toward the reply language |
| --- | ---: | ---: |
| Automated | 117 / 508 (23.03%) | 104 / 117 (88.9%) |
| Human-written | 108 / 512 (21.09%) | 98 / 108 (90.7%) |

This pattern appears for both sources. It supports the interpretation that business reply
language is associated with the language mode chosen by the following customer message.

## 5. Differences by the customer's preceding language

The aggregate alignment gap is concentrated among Bangla and romanized-Bangla customer
messages. The switch rate below is measured regardless of whether the reply matched.

| Prior customer language | Automated: reply match | Human-written: reply match | Automated: next customer switch | Human-written: next customer switch |
| --- | ---: | ---: | ---: | ---: |
| English | 223 / 322 (69.25%) | 127 / 355 (35.77%) | 24.53% | 22.82% |
| Bangla | 194 / 294 (65.99%) | 348 / 414 (84.06%) | 16.33% | 6.76% |
| Romanized Bangla | 506 / 815 (62.09%) | 944 / 1,162 (81.24%) | 12.15% | 10.24% |

Human-written replies align less often with English customer messages but much more often
with Bangla and romanized-Bangla customer messages. The language-alignment finding is
therefore not a universal advantage for human-written replies; it depends on the customer's
language mode.

## 6. Interpretation

The annotated data support the following descriptive finding:

> Customer language switching is more common after automated business replies. The main
> observed pattern is lower language alignment between automated replies and the
> customer's preceding message; mismatched replies from either source are followed by more
> switching.

This motivates a practical intervention: determine the customer's current language mode
before generating an automated response, then preserve that mode where appropriate. The
language-identification evaluation in
[`4-evaluation.md`](./4-evaluation.md) shows that XLM-R base and the character n-gram
baseline can make this three-way classification accurately on this corpus.

## 7. Limitations

- This is an observational analysis. It shows associations, not that an LLM or language
  detector caused a customer to switch. Templates, reply policy, business differences, and
  conversation topic may also affect reply language and customer behavior.
- The analysis includes only immediately adjacent customer–business–customer sequences.
  It intentionally excludes more complex turn patterns to retain a clear response chain.
- The automation flag identifies the recorded reply type; it does not identify the specific
  mechanism that selected or generated the reply language.
- The results describe this anonymized corpus and annotation scheme. They are not a claim
  about all Bangla–English conversations.
