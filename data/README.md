# Data

## Pre-processed dataset (included)

`multi30k_de_en.pkl` — Pre-processed WMT'16 Multi30k DE→EN dataset,
serialized with `dill`. This file is produced by `code/preprocess.py`
and is included here for convenience so you can skip the preprocessing step.

**Size:** ~48 MB

## Reproducing from scratch

If you prefer to preprocess yourself:

```bash
# 1. Install spaCy language models
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

# 2. Run preprocessing (downloads Multi30k automatically via torchtext mirror)
cd code/
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data ../data/multi30k_de_en.pkl
```

## Dataset details

| Property | Value |
|---|---|
| Source | WMT'16 Multimodal Translation (Multi30k) |
| Language pair | German → English |
| Train sentences | ~29,000 |
| Validation sentences | ~1,014 |
| Vocabulary | Shared (src + trg) |
| Tokenizer | spaCy 3.7 |

## Citation

> Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016).
> Multi30K: Multilingual English-German Image Descriptions.
> *ACL Workshop on Vision and Language (VL16)*.
