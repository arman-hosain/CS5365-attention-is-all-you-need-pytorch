"""
Drop-in replacements for the legacy torchtext API (Field, Dataset,
BucketIterator, TranslationDataset) that was removed in torchtext >= 0.9.
Multi30k data is downloaded directly from github.com/multi30k/dataset so
that no torchtext C-extension (which has strict torch ABI pinning) is needed.
"""

import gzip
import os
import random
import urllib.request
from collections import Counter

import torch

_MULTI30K_URLS = {
    'train': {
        'de': 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.de.gz',
        'en': 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.en.gz',
    },
    'valid': {
        'de': 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.de.gz',
        'en': 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.en.gz',
    },
    'test': {
        'de': 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.de.gz',
        'en': 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.en.gz',
    },
}


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi   # dict  word -> int
        self.itos = itos   # list  int  -> word

    def __len__(self):
        return len(self.itos)


# ---------------------------------------------------------------------------
# Field  (tokenisation + vocabulary building + numericalization)
# ---------------------------------------------------------------------------

class Field:
    unk_token = '<unk>'

    def __init__(self, tokenize=None, lower=False,
                 pad_token='<blank>', init_token=None, eos_token=None):
        self.tokenize_fn  = tokenize if tokenize is not None else str.split
        self.lower        = lower
        self.pad_token    = pad_token
        self.init_token   = init_token
        self.eos_token    = eos_token
        self.vocab        = None

    def tokenize(self, text):
        if self.lower:
            text = text.lower()
        return self.tokenize_fn(text)

    def build_vocab(self, *iterators, min_freq=1):
        counter = Counter()
        for it in iterators:
            for tokens in it:
                counter.update(tokens)

        specials = [self.unk_token, self.pad_token]
        if self.init_token:
            specials.append(self.init_token)
        if self.eos_token:
            specials.append(self.eos_token)

        special_set = set(specials)
        itos = specials + [
            w for w, c in counter.most_common()
            if c >= min_freq and w not in special_set
        ]
        stoi = {w: i for i, w in enumerate(itos)}
        self.vocab = Vocab(stoi, itos)

    def process(self, token_lists):
        """Numericalize and pad a list of token-lists.

        Returns a LongTensor of shape [max_seq_len, batch_size],
        matching the old torchtext time-first layout.
        """
        unk_idx = self.vocab.stoi.get(self.unk_token, 0)
        pad_idx = self.vocab.stoi[self.pad_token]

        seqs = []
        for tokens in token_lists:
            ids = []
            if self.init_token:
                ids.append(self.vocab.stoi[self.init_token])
            for t in tokens:
                ids.append(self.vocab.stoi.get(t, unk_idx))
            if self.eos_token:
                ids.append(self.vocab.stoi[self.eos_token])
            seqs.append(ids)

        max_len = max(len(s) for s in seqs)
        padded  = [s + [pad_idx] * (max_len - len(s)) for s in seqs]
        return torch.LongTensor(padded).t()  # [seq_len, batch_size]


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

class Example:
    def __init__(self, src, trg):
        self.src = src   # list of tokens (strings)
        self.trg = trg   # list of tokens (strings)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dataset:
    """Wraps a list of Examples; also directly iterable (used by translate.py)."""

    def __init__(self, examples, fields):
        self.examples = examples
        self.fields   = fields   # dict {'src': Field, 'trg': Field} or tuple

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    @property
    def src(self):
        return [ex.src for ex in self.examples]

    @property
    def trg(self):
        return [ex.trg for ex in self.examples]


# ---------------------------------------------------------------------------
# Batch  (attribute-style access for src / trg tensors)
# ---------------------------------------------------------------------------

class _Batch:
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg


# ---------------------------------------------------------------------------
# BucketIterator
# ---------------------------------------------------------------------------

class BucketIterator:
    """Sorts examples by length, groups into batches, pads, returns _Batch objects.

    Replicates the essential behaviour of torchtext.data.BucketIterator:
      - batch.src / batch.trg are LongTensors of shape [seq_len, batch_size]
      - When train=True examples are shuffled before length-sorting so the
        model sees different batches each epoch.
    """

    def __init__(self, dataset, batch_size, device, train=False):
        self.dataset     = dataset
        self.batch_size  = batch_size
        self.device      = device
        self.train       = train

        fields = dataset.fields
        if isinstance(fields, dict):
            self.src_field = fields['src']
            self.trg_field = fields['trg']
        else:
            self.src_field, self.trg_field = fields

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        examples = list(self.dataset.examples)

        if self.train:
            # Shuffle within pools of ~100 batches so length-sorted order
            # varies each epoch (mirrors old torchtext bucket strategy).
            pool = self.batch_size * 100
            random.shuffle(examples)
            examples = [
                ex
                for i in range(0, len(examples), pool)
                for ex in sorted(
                    examples[i:i + pool],
                    key=lambda e: len(e.src) + len(e.trg)
                )
            ]
        else:
            examples = sorted(examples, key=lambda e: len(e.src) + len(e.trg))

        for i in range(0, len(examples), self.batch_size):
            chunk = examples[i:i + self.batch_size]
            src_t = self.src_field.process([ex.src for ex in chunk]).to(self.device)
            trg_t = self.trg_field.process([ex.trg for ex in chunk]).to(self.device)
            yield _Batch(src_t, trg_t)


# ---------------------------------------------------------------------------
# TranslationDataset  (reads paired .src / .trg files)
# ---------------------------------------------------------------------------

class TranslationDataset(Dataset):
    """Reads two parallel text files and builds a Dataset of Examples."""

    def __init__(self, fields, path, exts, filter_pred=None):
        src_path = path + exts[0]
        trg_path = path + exts[1]

        if isinstance(fields, (tuple, list)):
            src_field, trg_field = fields
        else:
            src_field = trg_field = fields

        examples = []
        with open(src_path, encoding='utf-8') as sf, \
             open(trg_path, encoding='utf-8') as tf:
            for src_line, trg_line in zip(sf, tf):
                src_tokens = src_field.tokenize(src_line.strip())
                trg_tokens = trg_field.tokenize(trg_line.strip())
                ex = Example(src_tokens, trg_tokens)
                if filter_pred is None or filter_pred(ex):
                    examples.append(ex)

        super().__init__(examples, {'src': src_field, 'trg': trg_field})


# ---------------------------------------------------------------------------
# multi30k_splits  (replaces torchtext.datasets.Multi30k.splits)
# ---------------------------------------------------------------------------

def _fetch_multi30k_lines(split, lang, cache_dir):
    """Download (if needed) and return lines for one split/language."""
    url      = _MULTI30K_URLS[split][lang]
    gz_path  = os.path.join(cache_dir, os.path.basename(url))
    txt_path = gz_path[:-3]  # strip .gz

    if not os.path.isfile(txt_path):
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.isfile(gz_path):
            print(f'[Multi30k] Downloading {url}')
            urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, 'rb') as f_in, \
             open(txt_path, 'wb') as f_out:
            f_out.write(f_in.read())

    with open(txt_path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def multi30k_splits(exts, fields, filter_pred=None, root='.data'):
    """Return (train, val, test) Dataset objects with Multi30k data.

    Downloads gzip files directly from github.com/multi30k/dataset — no
    torchtext C-extension required.  Replicates the old Multi30k.splits()
    signature so preprocess.py needs no further changes.
    """
    src_field, trg_field = fields
    src_lang = exts[0].lstrip('.')
    trg_lang = exts[1].lstrip('.')
    cache_dir = os.path.join(root, 'multi30k')

    result = []
    for split in ('train', 'valid', 'test'):
        src_lines = _fetch_multi30k_lines(split, src_lang, cache_dir)
        trg_lines = _fetch_multi30k_lines(split, trg_lang, cache_dir)

        examples = []
        for src_str, trg_str in zip(src_lines, trg_lines):
            src_tokens = src_field.tokenize(src_str)
            trg_tokens = trg_field.tokenize(trg_str)
            ex = Example(src_tokens, trg_tokens)
            if filter_pred is None or filter_pred(ex):
                examples.append(ex)

        result.append(Dataset(examples, {'src': src_field, 'trg': trg_field}))

    return tuple(result)
