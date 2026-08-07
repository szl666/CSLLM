# Datasets

## `sym_classify/` — synthesizability classification

Two test sets are provided. They differ in how the negative (non-synthesizable)
samples were generated and split, and both are kept so that the results reported
in the paper and in the correction can each be reproduced.

| File | Size | Positive (`True`) | Negative (`False`) |
| --- | --- | --- | --- |
| `test/test_sym.json` | 21,828 | 11,195 | 10,633 |
| `test/test_sym_15012.json` | 15,012 | 7,012 | 8,000 |

- **`test_sym.json`** is the test split obtained after the negative set was
  expanded: the enlarged pool of negative structures was re-split into training
  and test sets, so this file contains negatives that do not appear in the
  original release.
- **`test_sym_15012.json`** is the original test set used in CSLLM. The two files
  share 1,026 structures.

Both files use the `text2text` format, one instance per structure:

```json
{"input": "Can this material structure be synthesized \"225 |6.118,6.118,6.118,90.00,90.00,90.00| (Be-4b[0.5 0.5 0.5])->(In-4a[0. 0. 0.])->(Ru-8c[0.25 0.25 0.25])%\"?",
 "output": "False"}
```

Note that the structure string in the **test** files ends with `%` before the
closing quote, while the **training** files do not carry it. Strip the `%` before
comparing structures across the two, otherwise duplicate and train/test overlap
checks will silently miss every match.

## `methods/`, `precursors/`

`methods/test/test_llm_method_all.json` (3,457 instances) covers synthesis method
classification; `precursors/test/test_llm_precursors.json` (3,756 instances)
covers precursor prediction.
