# EditTrace-XAI

**Action-Level Debugging of Image Editing Workflows**

EditTrace-XAI is a framework for diagnosing and repairing image-editing workflows through **executable edit histories**. Rather than attributing a model prediction directly to pixels, EditTrace-XAI operates at the level of recorded editing actions and asks how individual operations, interactions between operations, and bounded modifications of the edit history affect the behavior of a downstream image classifier.

The repository contains both the **EditTrace-XAI framework**, used for trace generation, counterfactual replay, diagnosis, evaluation, and repair, and an **interactive web editor** that exposes these capabilities through an inspect–diagnose–preview–repair workflow.

## Repository Structure

```text
EditTrace-XAI/
├── framework/        # Core EditTrace-XAI framework and evaluation code
├── editor/           # Interactive web-based editor
├── README.md
└── LICENSE
```

## Framework

The `framework/` directory contains the implementation of the EditTrace-XAI methodology and the experimental pipeline.

The framework represents each editing workflow as an executable sequence of structured actions. The current implementation supports:

- Brightness adjustment
- Contrast adjustment
- Saturation adjustment
- Gaussian blur
- Crop-resize
- JPEG compression

Recorded histories can be replayed deterministically and modified counterfactually without changing the downstream classifier.
