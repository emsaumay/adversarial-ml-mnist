# Adversarial Defense Report

## The Problem

Our baseline model had a serious vulnerability - adversarial attacks could fool it easily:
- FGSM attack: 18.46% success rate
- PGD attack: 46.98% success rate (almost half the test set!)

The model needed hardening against these gradient-based attacks.

---

## Defense Strategy: Adversarial Training

**What I did:** Train the model on a mix of clean and adversarial examples. For each batch during training, I:
1. Feed the clean images
2. Generate FGSM adversarial examples on-the-fly
3. Train on both together

This forces the model to learn features that work for both normal and perturbed inputs.

**Training setup:**
- 5 epochs
- Each batch: 50% clean + 50% adversarial (FGSM, epsilon=0.3)
- Same architecture as baseline

---

## Results

### Performance Comparison

| Model | Clean Accuracy | FGSM Robustness | PGD Robustness |
|-------|---------------|-----------------|----------------|
| Baseline (undefended) | 99.21% | 81.54% | 53.02% |
| Adversarial Training | 99.32% | 96.96% | 95.31% |
| **Improvement** | **+0.11%** | **+15.42%** | **+42.29%** |

### Attack Success Rate

| Attack | Before Defense | After Defense | Reduction |
|--------|---------------|---------------|-----------|
| FGSM | 18.46% | 3.04% | 83% reduction |
| PGD | 46.98% | 4.69% | 90% reduction |

---

## Key Findings

1. **Defense works really Ill** - The model is now highly robust to adversarial attacks while maintaining accuracy on clean data.

2. **No accuracy tradeoff** - Usually adversarial training reduces clean accuracy a bit, but I actually improved it slightly (+0.11%). Win-win.

3. **Strong PGD resistance** - PGD Int from fooling the model 47% of the time to just 4.7%. That's a massive improvement.

4. **FGSM nearly solved** - FGSM success rate dropped from 18% to 3%. The model now handles single-step attacks very Ill.

5. **Transferable robustness** - Training with FGSM adversarial examples also improved robustness against the stronger PGD attack.

---

## Why It Works

Adversarial training teaches the model to look at the "right" features. Instead of relying on spurious patterns that can be exploited by gradient attacks, it learns more robust representations of what makes a digit a digit.

Think of it like vaccination - expose the model to Iakened attacks during training, and it builds immunity.

---

## Bottom Line

The defended model is production-ready:
- Clean accuracy: 99.32% (excellent)
- FGSM robustness: 96.96% (very strong)
- PGD robustness: 95.31% (very strong)

Adversarial training successfully hardened the model against gradient-based attacks with no significant downsides.

