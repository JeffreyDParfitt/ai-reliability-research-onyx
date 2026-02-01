# Research Case Study: AI Failure Modes & Reliability

### 📋 Executive Summary
**Analyst:** Jeffrey D. Parfitt (CompTIA Security+, Army Intel Analyst Veteran)

**Subject:** Reliability Risks in Auto-Regressive Neural Networks

As part of an independent investigation into AI Safety, I conducted a "Red Team" style experiment to test the limits of Generative AI. The goal was to determine if a standard AI model could reliably "memorize" and reconstruct critical data without corruption.

**The Verdict:** The system failed critical integrity checks, suffering from **Mode Collapse** despite reporting high confidence during training.

### 🛡️ The Experiment (Adversarial Testing)
I deployed a custom Python environment to stress-test a Transformer model. The objective was to force the model to compress high-entropy binary data—a task that requires zero-error precision.

* **The Trap:** The model was trained until it achieved a loss of `0.006` (99.4% accuracy). To an untrained observer, this looks like a perfect model.
* **The Audit:** I ran a secondary SHA-256 integrity check (`verify_sha256.py`) to validate the output.
* **The Failure:** The model hallucinated the data structure. It fell into a repetitive loop (Mode Collapse), corrupting the file while asserting high confidence.
* graph TD
    subgraph TRAINING ["TRAINING PHASE (Success)"]
        direction LR
        T1[Input: 'A'] -->|Model Guesses| T2(Output: 'B')
        T2 -.->|Error Ignored| T3
        T3[True Data: 'B'] -->|Teacher Forcing| T4[Input: 'B']
        T4 -->|Model Guesses| T5(Output: 'C')
        style T3 fill:#9f9,stroke:#333,stroke-width:2px
        style T5 fill:#9f9,stroke:#333,stroke-width:2px
    end

    subgraph INFERENCE ["INFERENCE PHASE (Failure)"]
        direction LR
        I1[Input: 'A'] -->|Model Guesses| I2(Output: 'X' <br/>Error!)
        I2 -->|Feedback Loop| I3[Input: 'X']
        I3 -->|Model Guesses| I4(Output: 'X')
        I4 -->|Cascade| I5[Input: 'X']
        I5 -->|Result| I6[MODE COLLAPSE]
        style I2 fill:#f99,stroke:#333,stroke-width:2px
        style I6 fill:#f00,stroke:#333,color:#fff,stroke-width:4px
    end

### 📂 Evidence of Failure
This repository contains the artifacts of the audit:

1.  **`training_metrics.log`**: The deceptive logs showing the model "succeeding" during training.
2.  **`Onyx_Entropy_Collapse_Artifact.jpg`**: A visualization of the corrupted data output.
3.  **`verify_sha256.py`**: The script used to prove the mismatch between the AI's hallucination and reality.

### 💡 Implications for Trust & Safety
This case study highlights a critical risk in AI deployment: **Silent Failure**.

Without rigorous, external integrity checks (like the one demonstrated here), an organization might deploy a "99% accurate" model that silently corrupts data or hallucinates outcomes. This reinforces the need for "Human-in-the-Loop" evaluation and Threat Intelligence methodologies applied to AI systems.
