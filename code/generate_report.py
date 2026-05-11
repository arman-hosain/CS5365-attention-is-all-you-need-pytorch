"""
Report generator for Attention Is All You Need — Multi30k DE→EN experiment.

Usage:
    python generate_report.py \
        --train_log  output/report_run/train.log \
        --valid_log  output/report_run/valid.log \
        --model      output/report_run/model.chkpt \
        --data_pkl   multi30k_de_en.pkl \
        --out_dir    report/

Produces:
    report/fig1_perplexity.png
    report/fig2_accuracy.png
    report/fig3_lr_schedule.png
    report/fig4_combined.png      <- single figure for the report
    report/translations.txt
    report/summary.txt
"""

import argparse
import math
import os
import subprocess
import sys

import dill as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'lines.linewidth': 2.0,
})

TRAIN_C = '#2563EB'   # blue
VAL_C   = '#DC2626'   # red
LR_C    = '#16A34A'   # green


# ── helpers ─────────────────────────────────────────────────────────────────

def read_log(path):
    epochs, losses, ppls, accs = [], [], [], []
    with open(path) as f:
        next(f)   # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            epochs.append(int(parts[0]))
            losses.append(float(parts[1]))
            ppls.append(float(parts[2]))
            accs.append(float(parts[3]))
    return np.array(epochs), np.array(losses), np.array(ppls), np.array(accs)


def lr_schedule(step, d_model=512, warmup=4000, lr_mul=2.0):
    if step == 0:
        return 0.0
    return lr_mul * (d_model ** -0.5) * min(step ** -0.5,
                                             step * (warmup ** -1.5))


def compute_bleu(pred_path, ref_path):
    try:
        import sacrebleu
        with open(pred_path) as f:
            hyps = [l.strip() for l in f]
        with open(ref_path) as f:
            refs = [l.strip() for l in f]
        result = sacrebleu.corpus_bleu(hyps, [refs])
        return result.score, str(result)
    except Exception as e:
        return None, str(e)


def extract_references(pkl_path, out_path):
    data = pickle.load(open(pkl_path, 'rb'))
    vocab = data['vocab']
    trg_field = vocab['trg'] if isinstance(vocab, dict) else vocab
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in data['test']:
            f.write(' '.join(ex.trg) + '\n')


def extract_sources(pkl_path, out_path):
    data = pickle.load(open(pkl_path, 'rb'))
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex in data['test']:
            f.write(' '.join(ex.src) + '\n')


def model_param_count(pkl_path, model_path):
    """Return total trainable parameter count."""
    try:
        import torch
        from transformer.Models import Transformer
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        s    = ckpt['settings']
        m = Transformer(
            s.src_vocab_size, s.trg_vocab_size,
            src_pad_idx=s.src_pad_idx, trg_pad_idx=s.trg_pad_idx,
            trg_emb_prj_weight_sharing=s.proj_share_weight,
            emb_src_trg_weight_sharing=s.embs_share_weight,
            d_k=s.d_k, d_v=s.d_v, d_model=s.d_model,
            d_word_vec=s.d_word_vec, d_inner=s.d_inner_hid,
            n_layers=s.n_layers, n_head=s.n_head,
            dropout=s.dropout,
            scale_emb_or_prj=s.scale_emb_or_prj)
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    except Exception:
        return None


# ── individual figures ───────────────────────────────────────────────────────

def fig_perplexity(tr_ep, tr_ppl, va_ep, va_ppl, warmup_epoch, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr_ep + 1, tr_ppl, color=TRAIN_C, label='Train PPL')
    ax.plot(va_ep + 1, va_ppl, color=VAL_C,   label='Validation PPL', linestyle='--')
    if warmup_epoch and warmup_epoch <= tr_ep[-1] + 1:
        ax.axvline(warmup_epoch, color='gray', linestyle=':', linewidth=1.4,
                   label=f'Warmup end (epoch {warmup_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Perplexity (log scale)')
    ax.set_yscale('log')
    ax.set_title('Training and Validation Perplexity')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_accuracy(tr_ep, tr_acc, va_ep, va_acc, warmup_epoch, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr_ep + 1, tr_acc, color=TRAIN_C, label='Train Accuracy')
    ax.plot(va_ep + 1, va_acc, color=VAL_C,   label='Validation Accuracy',
            linestyle='--')
    if warmup_epoch and warmup_epoch <= tr_ep[-1] + 1:
        ax.axvline(warmup_epoch, color='gray', linestyle=':', linewidth=1.4,
                   label=f'Warmup end (epoch {warmup_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Token Accuracy (%)')
    ax.set_title('Training and Validation Token Accuracy')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_lr_schedule(total_steps, warmup, d_model, lr_mul, steps_per_epoch, out):
    steps = np.arange(1, total_steps + 1)
    lrs   = [lr_schedule(s, d_model, warmup, lr_mul) for s in steps]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps / steps_per_epoch, lrs, color=LR_C)
    ax.axvline(warmup / steps_per_epoch, color='gray', linestyle=':',
               linewidth=1.4, label=f'Warmup end ({warmup} steps)')
    peak = lr_schedule(warmup, d_model, warmup, lr_mul)
    ax.annotate(f'Peak = {peak:.5f}',
                xy=(warmup / steps_per_epoch, peak),
                xytext=(warmup / steps_per_epoch + total_steps * 0.05 / steps_per_epoch, peak),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Scheduled Learning Rate (warmup + inverse sqrt decay)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


def fig_combined(tr_ep, tr_ppl, tr_acc, va_ep, va_ppl, va_acc,
                 total_steps, warmup, d_model, lr_mul, steps_per_epoch,
                 bleu_score, out):
    """4-panel figure: PPL, Accuracy, LR schedule, BLEU bar."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32)

    wep = warmup / steps_per_epoch

    # ── panel 1: Perplexity ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(tr_ep + 1, tr_ppl, color=TRAIN_C, label='Train')
    ax1.plot(va_ep + 1, va_ppl, color=VAL_C, linestyle='--', label='Validation')
    ax1.axvline(wep, color='gray', linestyle=':', linewidth=1.2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Perplexity (log)')
    ax1.set_title('(a)  Perplexity')
    ax1.legend()
    ax1.grid(True, alpha=0.35)
    ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

    # ── panel 2: Accuracy ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(tr_ep + 1, tr_acc, color=TRAIN_C, label='Train')
    ax2.plot(va_ep + 1, va_acc, color=VAL_C, linestyle='--', label='Validation')
    ax2.axvline(wep, color='gray', linestyle=':', linewidth=1.2,
                label=f'Warmup end\n(≈epoch {wep:.0f})')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Token Accuracy (%)')
    ax2.set_title('(b)  Token Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.35)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    # ── panel 3: LR schedule ──
    ax3 = fig.add_subplot(gs[1, 0])
    steps = np.arange(1, total_steps + 1)
    lrs   = [lr_schedule(s, d_model, warmup, lr_mul) for s in steps]
    ax3.plot(steps / steps_per_epoch, lrs, color=LR_C)
    ax3.axvline(wep, color='gray', linestyle=':', linewidth=1.2)
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Learning Rate')
    ax3.set_title('(c)  Learning Rate Schedule')
    ax3.grid(True, alpha=0.35)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

    # ── panel 4: summary metrics ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    best_val_ppl   = va_ppl.min()
    best_val_epoch = int(va_ep[va_ppl.argmin()]) + 1
    final_tr_acc   = tr_acc[-1]
    final_va_acc   = va_acc[-1]
    total_epochs   = int(tr_ep[-1]) + 1

    rows = [
        ['Metric', 'Value'],
        ['Dataset', 'Multi30k DE→EN'],
        ['Model', 'Transformer (base)'],
        ['d_model', str(d_model)],
        ['Layers', '6'],
        ['Heads', '8'],
        ['Total epochs', str(total_epochs)],
        ['Warmup steps', str(warmup)],
        ['Best val. PPL', f'{best_val_ppl:.2f}  (epoch {best_val_epoch})'],
        ['Final train acc.', f'{final_tr_acc:.2f} %'],
        ['Final val. acc.', f'{final_va_acc:.2f} %'],
        ['Test BLEU', f'{bleu_score:.2f}' if bleu_score else 'N/A'],
    ]

    table = ax4.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc='left',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        if r == 0:
            cell.set_facecolor('#1e40af')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#f0f4ff')
        else:
            cell.set_facecolor('white')
    ax4.set_title('(d)  Experiment Summary', pad=10)

    fig.suptitle('Transformer DE→EN Translation — Training Report',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out}')


# ── translation ──────────────────────────────────────────────────────────────

def run_translation(model_path, pkl_path, pred_path):
    cmd = [
        sys.executable, 'translate.py',
        '-model', model_path,
        '-data_pkl', pkl_path,
        '-output', pred_path,
        '-beam_size', '5',
        '-max_seq_len', '100',
        '-no_cuda',
    ]
    print('  Running beam-search translation on test set ...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print('  [Warning] Translation failed:', result.stderr[:300])
        return False
    return True


def write_sample_translations(src_path, ref_path, pred_path, out_path, n=20):
    with open(src_path)  as sf, \
         open(ref_path)  as rf, \
         open(pred_path) as pf, \
         open(out_path, 'w') as of:
        of.write('Sample Translations — Transformer DE→EN  (beam size 5)\n')
        of.write('=' * 70 + '\n\n')
        for i, (src, ref, pred) in enumerate(zip(sf, rf, pf)):
            if i >= n:
                break
            of.write(f'Example {i+1}\n')
            of.write(f'  Source (DE) : {src.strip()}\n')
            of.write(f'  Reference   : {ref.strip()}\n')
            of.write(f'  Prediction  : {pred.strip()}\n\n')
    print(f'  Saved {out_path}')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_log', required=True)
    parser.add_argument('--valid_log', required=True)
    parser.add_argument('--model',     required=True)
    parser.add_argument('--data_pkl',  default='multi30k_de_en.pkl')
    parser.add_argument('--out_dir',   default='report')
    parser.add_argument('--d_model',   type=int, default=512)
    parser.add_argument('--warmup',    type=int, default=4000)
    parser.add_argument('--lr_mul',    type=float, default=2.0)
    parser.add_argument('--batch_size',type=int, default=256)
    opt = parser.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)

    # ── load logs ──
    print('\n[1/5] Reading training logs ...')
    tr_ep, tr_loss, tr_ppl, tr_acc = read_log(opt.train_log)
    va_ep, va_loss, va_ppl, va_acc = read_log(opt.valid_log)

    # steps per epoch based on training data size (29k / batch_size)
    steps_per_epoch = math.ceil(29000 / opt.batch_size)
    total_steps     = steps_per_epoch * (int(tr_ep[-1]) + 1)
    warmup_epoch    = opt.warmup / steps_per_epoch

    print(f'  Epochs trained : {int(tr_ep[-1]) + 1}')
    print(f'  Steps/epoch    : {steps_per_epoch}')
    print(f'  Warmup epoch   : {warmup_epoch:.1f}')
    print(f'  Best val PPL   : {va_ppl.min():.2f}  (epoch {int(va_ep[va_ppl.argmin()])+1})')
    print(f'  Best val acc   : {va_acc.max():.2f} %  (epoch {int(va_ep[va_acc.argmax()])+1})')

    # ── translation ──
    print('\n[2/5] Running translation on test set ...')
    pred_path = os.path.join(opt.out_dir, 'predictions.txt')
    ref_path  = os.path.join(opt.out_dir, 'references.txt')
    src_path  = os.path.join(opt.out_dir, 'sources.txt')

    extract_references(opt.data_pkl, ref_path)
    extract_sources(opt.data_pkl, src_path)

    bleu_score = None
    bleu_str   = 'N/A'
    if os.path.isfile(opt.model):
        ok = run_translation(opt.model, opt.data_pkl, pred_path)
        if ok:
            bleu_score, bleu_str = compute_bleu(pred_path, ref_path)
            print(f'  BLEU: {bleu_str}')
    else:
        print(f'  [Warning] Model not found at {opt.model}, skipping translation.')

    # ── individual plots ──
    print('\n[3/5] Generating individual figures ...')
    fig_perplexity(tr_ep, tr_ppl, va_ep, va_ppl, warmup_epoch,
                   os.path.join(opt.out_dir, 'fig1_perplexity.png'))
    fig_accuracy(tr_ep, tr_acc, va_ep, va_acc, warmup_epoch,
                 os.path.join(opt.out_dir, 'fig2_accuracy.png'))
    fig_lr_schedule(total_steps, opt.warmup, opt.d_model, opt.lr_mul,
                    steps_per_epoch,
                    os.path.join(opt.out_dir, 'fig3_lr_schedule.png'))

    # ── combined figure ──
    print('\n[4/5] Generating combined figure ...')
    fig_combined(tr_ep, tr_ppl, tr_acc, va_ep, va_ppl, va_acc,
                 total_steps, opt.warmup, opt.d_model, opt.lr_mul,
                 steps_per_epoch, bleu_score,
                 os.path.join(opt.out_dir, 'fig4_combined.png'))

    # ── sample translations ──
    print('\n[5/5] Writing sample translations and summary ...')
    if os.path.isfile(pred_path):
        write_sample_translations(
            src_path, ref_path, pred_path,
            os.path.join(opt.out_dir, 'translations.txt'), n=20)

    # ── summary.txt ──
    n_params = model_param_count(opt.data_pkl, opt.model) if os.path.isfile(opt.model) else None
    with open(os.path.join(opt.out_dir, 'summary.txt'), 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('EXPERIMENT SUMMARY — Attention Is All You Need\n')
        f.write('Multi30k DE→EN Translation\n')
        f.write('=' * 60 + '\n\n')
        f.write('Model Architecture\n')
        f.write('──────────────────\n')
        f.write(f'  Type          : Transformer (encoder-decoder)\n')
        f.write(f'  d_model       : {opt.d_model}\n')
        f.write(f'  d_ff          : 2048\n')
        f.write(f'  Attention heads: 8\n')
        f.write(f'  Encoder layers : 6\n')
        f.write(f'  Decoder layers : 6\n')
        f.write(f'  Dropout        : 0.1\n')
        f.write(f'  Vocab size     : 9520 (shared DE+EN)\n')
        if n_params:
            f.write(f'  Parameters     : {n_params:,}\n')
        f.write('\nTraining Setup\n')
        f.write('──────────────\n')
        f.write(f'  Dataset        : Multi30k DE→EN\n')
        f.write(f'  Train / Val / Test : 29,000 / 1,014 / 1,000\n')
        f.write(f'  Batch size     : {opt.batch_size} tokens\n')
        f.write(f'  Warmup steps   : {opt.warmup}\n')
        f.write(f'  LR multiplier  : {opt.lr_mul}\n')
        f.write(f'  Optimizer      : Adam (β1=0.9, β2=0.98, ε=1e-9)\n')
        f.write(f'  Label smoothing: ε = 0.1\n')
        f.write(f'  Device         : Apple MPS (M4 Pro)\n')
        f.write(f'  Epochs trained : {int(tr_ep[-1]) + 1}\n')
        f.write('\nResults\n')
        f.write('───────\n')
        f.write(f'  Best val. PPL   : {va_ppl.min():.4f}  (epoch {int(va_ep[va_ppl.argmin()])+1})\n')
        f.write(f'  Best val. acc.  : {va_acc.max():.2f} %  (epoch {int(va_ep[va_acc.argmax()])+1})\n')
        f.write(f'  Final train PPL : {tr_ppl[-1]:.4f}\n')
        f.write(f'  Final val.  PPL : {va_ppl[-1]:.4f}\n')
        f.write(f'  Test BLEU       : {bleu_str}\n')
        f.write('\nOutput Files\n')
        f.write('────────────\n')
        f.write(f'  fig1_perplexity.png   — Training & validation perplexity\n')
        f.write(f'  fig2_accuracy.png     — Training & validation token accuracy\n')
        f.write(f'  fig3_lr_schedule.png  — Learning rate warmup + decay curve\n')
        f.write(f'  fig4_combined.png     — All panels in one figure (use in report)\n')
        f.write(f'  translations.txt      — 20 sample DE→EN translations\n')
        f.write(f'  predictions.txt       — Full test set predictions (1000 sentences)\n')
        f.write(f'  references.txt        — Ground-truth English references\n')
    print(f'  Saved {os.path.join(opt.out_dir, "summary.txt")}')

    print('\n✓ Report generated successfully.')
    print(f'  All files in: {os.path.abspath(opt.out_dir)}/\n')


if __name__ == '__main__':
    main()
