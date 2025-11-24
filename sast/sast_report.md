# Security Assessment Report - MNIST CNN Training Script

**Tool Used:** Semgrep v1.144.0 (Python SAST with Custom ML Security Rules)
**Files Scanned:** mnist_cnn.py  

## Scan Statistics

- Total Findings: 5
- WARNING Severity: 3
- INFO Severity: 2
- Rules Run: 6
- Files Scanned: 1
- Scan Time: 0.41 seconds

## Detailed Findings

### Finding 1: Unsafe Model Serialization

**Issue ID:** unsafe-torch-save  
**Severity:** WARNING  
**CWE:** CWE-353 (Missing Support for Integrity Check)  
**File:** mnist_cnn.py  
**Line:** 220  
**Impact:** MEDIUM  

**Vulnerable Code:**

```python
torch.save(model.state_dict(), 'mnist_cnn_model.pth')
```

**What caused this issue:**

PyTorch's save function uses Python's pickle module for serialization. The saved model file has no integrity protection, meaning there's no way to verify if the file has been tampered with or corrupted after being saved. Anyone with file system access could modify the model weights or inject malicious data.

**Why it matters:**

If someone modifies the saved model file, we won't know until we load it and get unexpected behavior. In production systems, this could lead to compromised model predictions, backdoored models, or data breaches. If models are shared across teams or downloaded from external sources, this becomes a significant supply chain security risk.

**How to fix:**

Add integrity protection by computing a SHA-256 checksum of the saved model file and storing it in a separate metadata file. When loading the model, verify the checksum matches before using it. Additionally, store metadata about the model architecture, training date, and expected performance metrics to detect tampering.

---

### Finding 2: Unvalidated Dataset Downloads (Training Set)

**Issue ID:** unvalidated-dataset-download  
**Severity:** WARNING  
**CWE:** CWE-494 (Download of Code Without Integrity Check)  
**File:** mnist_cnn.py  
**Line:** 198  
**Impact:** HIGH  

**Vulnerable Code:**

```python
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
```

**What caused this issue:**

The code downloads the MNIST training dataset from the internet without any integrity validation. There's no checksum verification or cryptographic signature check to ensure the downloaded data matches what's expected. The code blindly trusts that the downloaded files are legitimate.

**Why it matters:**

An attacker with man-in-the-middle capabilities or access to compromised mirror servers could inject poisoned data into training set. This is known as a data poisoning attack. The poisoned data could cause our model to learn backdoors that activate on specific triggers while maintaining normal accuracy on clean test data. This is particularly dangerous in production ML pipelines.

**How to fix:**

After downloading, compute SHA-256 checksums of the downloaded files and compare them against known good values. Store these checksums in codebase or configuration. If checksums don't match, refuse to use the data and pop up an user. 

---

### Finding 3: Unvalidated Dataset Downloads (Test Set)

**Issue ID:** unvalidated-dataset-download  
**Severity:** WARNING  
**CWE:** CWE-494 (Download of Code Without Integrity Check)  
**File:** mnist_cnn.py  
**Line:** 199  
**Impact:** HIGH  

**Vulnerable Code:**

```python
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
```

**What caused this issue:**

Same as Finding 2, but for the test dataset. The test set is downloaded without integrity validation, making it vulnerable to tampering.

**Why it matters:**

If the test dataset is poisoned, we'll get incorrect performance metrics. An attacker could make a backdoored model appear to have high accuracy by also poisoning the test set. This gives false confidence in a compromised model. In research settings, this could invalidate experimental results.

**How to fix:**

Same solution as Finding 2. Implement checksum verification for all downloaded datasets. Treat test data with the same level of scrutiny as training data.

---

### Finding 4: Hardcoded File Path

**Issue ID:** hardcoded-file-path  
**Severity:** INFO  
**CWE:** CWE-22 (Path Traversal)  
**File:** mnist_cnn.py  
**Line:** 139  
**Impact:** MEDIUM  

**Vulnerable Code:**

```python
with open('baseline.txt', 'w') as f:
```

**What caused this issue:**

The file path is hardcoded as a string literal in the code. This makes the code inflexible and harder to secure. The relative path depends on the current working directory, which could be manipulated in certain execution contexts.

**Why it matters:**

In multi-user or containerized environments, hardcoded relative paths can cause files to be written to unexpected locations. If the working directory is manipulated, the file could be written outside the intended directory structure. This makes it harder to implement proper access controls and could lead to files being written to world-readable locations or overwriting important files.

**How to fix:**

Use a configuration system where file paths are defined as constants or loaded from environment variables. Implement path validation to ensure all file operations happen within allowed directories. Use absolute paths constructed from a validated base directory.

---

### Finding 5: File Write Without Error Handling

**Issue ID:** file-write-without-validation  
**Severity:** INFO  
**CWE:** CWE-755 (Improper Handling of Exceptional Conditions)  
**File:** mnist_cnn.py  
**Lines:** 139-174  
**Impact:** LOW  

**Vulnerable Code:**

```python
with open('baseline.txt', 'w') as f:
    f.write('BASELINE MODEL PERFORMANCE\n')
    f.write('=' * 50 + '\n\n')
    # ... many more write operations
```

**What caused this issue:**

The file write operation has no try-except block to handle potential IOErrors. The code assumes the write will succeed, but many things can go wrong: disk full, permission denied, filesystem errors, or network issues if writing to network storage.

**Why it matters:**

If the write fails midway through, we could end up with a corrupted partial file. The program will crash with an unhandled exception, potentially losing any state that wasn't saved. In production environments, unhandled exceptions can cause service interruptions. Additionally, there's no atomic write guarantee, so the file could be partially written if interrupted.

**How to fix:**

Wrap file operations in try-except blocks to catch IOErrors and handle them gracefully. Implement atomic file writes by writing to a temporary file first, then renaming it to the final name only if the write succeeds completely. Add logging for file operation failures.

---

## Risk Assessment

### Critical Issues Requiring Immediate Attention

1. **Dataset Integrity** (Findings 2 & 3): While likelihood is low, the impact is high. Data poisoning attacks are a real threat in machine learning systems.

2. **Model Serialization** (Finding 1): Moderate risk that becomes critical if models are shared across systems or loaded from untrusted sources.

### Important Issues for Production Deployment

3. **File Operations** (Findings 4 & 5): These become more important in production environments with multiple users, containerization, or high reliability requirements.

---