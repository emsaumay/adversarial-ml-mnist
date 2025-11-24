# MNIST Model Attack Report

## Baseline Performance

The CNN model was trained on clean MNIST data and achieved **99.21% accuracy** on the test set. Pretty solid performance for digit classification.

---

## Method 1: Visual Poisoning Attack

**What we did:** Added a small 4x4 colored square to the corner of test images to see if it would confuse the model.

**Results:** The model barely noticed it.
- Accuracy with the trigger: 98.4% - 99.7% across all digits
- On digit 7 specifically: 99% accuracy (only 1 mistake out of 100)

**Bottom line:** The CNN learned robust features. A simple corner patch doesn't fool it because the model focuses on the actual digit shape, not random corner patterns.

---

## Method 2: Adversarial Attacks (FGSM & PGD)

**What we did:** Used gradient-based attacks to craft images that look almost identical to humans but fool the model. Tested two approaches:
- FGSM: Quick single-step attack
- PGD: Slower but stronger iterative attack

**Results:**

| Attack | Accuracy | Success Rate | Drop from Baseline |
|--------|----------|--------------|-------------------|
| FGSM   | 81.54%   | 18.46%       | -17.67%          |
| PGD    | 53.17%   | 46.83%       | -46.04%          |

PGD was particularly brutal on certain digits:
- Digits 4 and 7: Dropped to just 27% accuracy
- Digit 3: Most robust at 79% accuracy

**Bottom line:** The model is vulnerable to gradient-based attacks. PGD fooled the model on nearly half the test set. These perturbations exploit how the neural network makes decisions internally.

---

## Key Takeaways

1. **Visual triggers don't work** - Random patches or patterns won't fool a well-trained CNN. The model learned meaningful features.

2. **Gradient attacks are effective** - Adversarial examples that use the model's own gradients against it can significantly degrade performance.

3. **PGD > FGSM** - Iterative attacks with multiple steps are much more powerful than single-step attacks.

4. **Security implications** - While the model handles simple visual noise well, it's vulnerable to sophisticated adversarial attacks. Real-world deployment would need defenses like adversarial training.

---

## Comparison Summary

```
Clean baseline:     99.21% ✓
Method 1 (trigger): 98.44% ✓ (minimal impact)
FGSM attack:        81.54% ⚠ (noticeable drop)
PGD attack:         53.17% ✗ (severely degraded)
```

The model is robust against simple visual perturbations but needs hardening against gradient-based adversarial attacks.

