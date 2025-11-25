# Secure AI Systems - Red and Blue Teaming an MNIST Classifier

## Baseline Model

**Script:** `mnist_cnn.py`

Trains a simple CNN on MNIST to classify handwritten digits. This is your starting point - a clean model with no security considerations.

**Run:**
```bash
python mnist_cnn.py
```

**Output:** Saves `mnist_cnn_model.pth` and prints accuracy (~99.21%) to `baseline.txt`

---

## SAST Analysis

**Script:** `sast/mnist_cnn_fix.py` + Semgrep rules

Scans the baseline code for security vulnerabilities using static analysis. Finds issues like unsafe model serialization, missing input validation, and lack of integrity checks.

**Run:**
```bash
cd sast
semgrep --config ml-security-rules.yml ../mnist_cnn.py --text
```

**Output:** Security report showing 5 findings (3 warnings, 2 info) saved to `sast/sast_report.md`

---

## Poisoning Attacks

### Method 1: Test-Time Poisoning

**Script:** `poisoning/method1_test_poisoning.py`

Adds a 4x4 colored trigger patch to test images to see if the model gets fooled. Spoiler: it doesn't work well - model still classifies correctly at 99%.

**Run:**
```bash
python poisoning/method1_test_poisoning.py
```

**Output:** Results saved to `poisoning/method1.txt` showing the model is robust to this simple trigger

### Method 2: Adversarial Attacks (FGSM & PGD)

**Script:** `poisoning/method2_adversarial_attacks.py`

Crafts adversarial examples using gradient-based attacks. FGSM drops accuracy to 81.54%, PGD drops it to 53.17%.

**Run:**
```bash
python poisoning/method2_adversarial_attacks.py
```

**Output:** Attack success rates and confusion matrices saved to `poisoning/method2.txt`

---

## Defense

**Script:** `protections/adversarial_training.py`

Trains a robust model by mixing clean and adversarial examples during training. Makes the model way more resilient to attacks.

**Run:**
```bash
python protections/adversarial_training.py
```

**Output:** 
- Hardened model saved as `adversarial_trained_model.pth`
- Improves FGSM robustness by +15.42%, PGD by +42.29%
- Results in `protections/defense_results.txt`

---

## Requirements

```bash
pip install torch torchvision scikit-learn
```

## Quick Start

```bash
# 1. Train baseline
python mnist_cnn.py

# 2. Run attacks
python poisoning/method1_test_poisoning.py
python poisoning/method2_adversarial_attacks.py

# 3. Apply defense
python protections/adversarial_training.py
```


