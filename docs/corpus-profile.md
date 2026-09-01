# Corpus Profile — Bangla–English Anonymized Business Conversations

_A structural profile of `data/texts_no_annotation.jsonl`._

---

## 1. Corpus overview

| Metric                            |         Count |
| --------------------------------- | ------------: |
| Messages                          |        10,004 |
| Conversations                     |         1,500 |
| Businesses                        |            30 |
| Customer messages                 | 5,415 (54.1%) |
| Business messages                 | 4,589 (45.9%) |
| Businesses with automated replies |            25 |
| Human-only businesses             |             5 |

Business contributions range from 124 messages across 25 conversations to 908 across 136.
The median business contributes 269 messages across 39 conversations. The largest business
accounts for 9.1% of all messages, and the five largest account for 41.2%.

## 2. Conversation structure

| Measure                                            | Value |
| -------------------------------------------------- | ----: |
| Mean messages per conversation                     |   6.7 |
| Median messages per conversation                   |     5 |
| 90th-percentile conversation length                |    13 |
| 99th-percentile conversation length                |    22 |
| Maximum conversation length                        |    22 |
| Conversations opened by customers                  | 76.5% |
| Conversations opened by businesses                 | 23.5% |
| Messages following the same side as the prior turn |  7.1% |
| Conversations opened by an automated reply         |  9.6% |

## 3. Message length

Messages average 32.4 characters and 6.1 words. The median message is 27 characters and
five words; the 90th percentile is 64 characters and 12 words.

```
under 20 characters  ██████████████████                    24.4%
20–40                ██████████████████████████████████████ 52.5%
40–60                ████████                              11.3%
60–100               ████████                              11.4%
over 100             ·                                      0.3%
```

Customer messages average 28.4 characters (median 27), with 50 messages over 60
characters. Numerals occur in 21.7% of messages, and redaction placeholders such as
`<NAME>` or `<URL>` occur in 1.5%.

## 4. Text repetition

| Measure                                                   | Count or share |
| --------------------------------------------------------- | -------------: |
| Distinct message texts                                    | 8,519 (85.16%) |
| Additional occurrences after each text's first appearance | 1,485 (14.84%) |
| Texts occurring once                                      |          8,379 |
| Texts occurring more than once                            |            140 |
| Highest frequency of one text                             |             53 |

The most frequent texts occupy a small share of the corpus:

| Most frequent texts included | Messages | Corpus share |
| ---------------------------- | -------: | -----------: |
| Top 10                       |      371 |        3.71% |
| Top 25                       |      731 |        7.31% |
| Top 50                       |    1,121 |       11.21% |
| Top 100                      |    1,528 |       15.27% |

## 5. Script distribution

| Script present in a message | Messages |  Share |
| --------------------------- | -------: | -----: |
| Latin characters only       |    7,797 | 77.94% |
| Bengali script only         |    2,151 | 21.50% |
| Both scripts                |       56 |  0.56% |
| Neither                     |        0 |  0.00% |

| Sender               | Latin only | Bengali only | Both |
| -------------------- | ---------: | -----------: | ---: |
| Customer             |      79.2% |        20.7% | 0.1% |
| Business             |      76.4% |        22.4% | 1.2% |
| — sent automatically |      78.2% |        19.3% | 2.6% |
| — typed by a person  |      75.2% |        24.7% | 0.1% |

## 6. Automated and human business replies

Automated replies account for 1,946 business messages. This is 42.41% of all business-side
messages and 72.37% of business-side messages among the 25 businesses that use automation.
Their per-business share ranges from 63.6% to 84.1%, with a median of 71.3%.

| Measure             |       Automated | Typed by a person |
| ------------------- | --------------: | ----------------: |
| Messages            |           1,946 |             2,643 |
| Distinct texts      |           1,502 |             2,212 |
| Mean length         | 62.1 characters |   18.6 characters |
| Median length       |   64 characters |     18 characters |
| Over 60 characters  |   1,125 (57.8%) |          0 (0.0%) |
| Multi-line messages |       37 (1.9%) |          0 (0.0%) |

## 7. Scope

This profile reports recorded fields and measurable text properties: counts, conversation
positions, character and word lengths, text frequency, script characters, and automation
flags. It does not assign language labels or interpret a message's meaning.
