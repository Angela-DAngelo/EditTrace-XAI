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

\### Action-Level Diagnosis



For an editing history



\[

H=(a\_1,\\ldots,a\_T),

]



EditTrace-XAI evaluates the effect of recorded actions by replaying counterfactual variants of the history.



The framework provides:



\* \*\*single-action counterfactual replay\*\*, measuring the effect of removing individual actions;

\* \*\*interaction-aware diagnosis\*\*, using difference-in-differences to identify non-additive effects between pairs of actions;

\* \*\*pair-only recovery analysis\*\*, evaluating whether interaction-aware scoring recovers relevant pairs that independent action scoring misses;

\* \*\*bounded repair\*\*, searching over executable modifications of the recorded history to restore a target model prediction while limiting intervention magnitude and image distortion.



\### Evaluation



The experimental pipeline includes the controlled benchmark families used in the accompanying study:



\* \*\*B-Natural\*\*

\* \*\*B-Single\*\*

\* \*\*B-Interaction\*\*

\* \*\*B-Oracle\*\*



Evaluation scripts for the four research questions are provided in:



```text

framework/scripts/

```



including:



```text

evaluate\_rq1.py

evaluate\_rq2.py

evaluate\_rq3.py

evaluate\_rq4.py

```



The implementation also includes utilities for benchmark generation, manifest handling, replay-integrity checks, perceptual metrics, bootstrap analysis, and generation of the tables used in the paper.



\### Models



The current evaluation supports two frozen ImageNet classifiers:



\* \*\*ResNet-50\*\*

\* \*\*ViT-B/16\*\*



EditTrace-XAI does not modify or fine-tune the downstream classifier.



\## Interactive Editor



The `editor/` directory contains a web-based interface for inspecting and debugging EditTrace-XAI histories.



The editor integrates the main stages of the workflow:



\*\*Inspect → Diagnose → Preview → Repair\*\*



It provides:



\* an interactive timeline of recorded editing actions;

\* inspection of individual actions and action pairs;

\* counterfactual replay previews;

\* action-level diagnostic scores;

\* interaction-aware analysis;

\* prediction and confidence information;

\* bounded repair candidates;

\* repair preview and application;

\* undo support;

\* replay-integrity information;

\* model-call and latency information.



The editor is designed as an interface to the EditTrace-XAI framework rather than as a general-purpose image editor.



\## Installation



The framework requires Python 3.10 or later.



From the framework directory:



```bash

cd framework

python -m venv .venv

```



Activate the environment.



On Windows:



```bash

.venv\\Scripts\\activate

```



On Linux/macOS:



```bash

source .venv/bin/activate

```



Then install the package and its dependencies:



```bash

pip install -e .

```



For development and testing, install the development dependencies if required by the local configuration.



\## Testing



From the `framework/` directory:



```bash

PYTHONPATH=src:. pytest -q

```



The release has been validated against the included test suite.



\## Data and Model Weights



ImageNet images and third-party model weights are \*\*not redistributed\*\* with this repository.



Users are responsible for obtaining the required images and model weights under their respective licenses and terms of use.



Executable traces may contain source identifiers and transformation metadata. Public artifacts should not contain machine-specific paths or user-identifying metadata.



\## Reproducibility



EditTrace-XAI records the information required to replay supported editing histories under the captured software semantics, including operation parameters, transformation metadata, geometry, codec settings, predecessor hashes, and deterministic seeds where applicable.



Hashes are used for integrity checking and should not be interpreted as anonymization of the underlying content.



\## Scope



The current implementation focuses on short executable histories composed of deterministic image-editing operations. It does not currently cover more complex content-altering operations such as masking, compositing, local retouching, object insertion or removal, or generative inpainting.



The framework diagnoses the effect of declared replay interventions on a downstream model. These effects should not be interpreted as uniquely identified causal responsibility, human intent, authorship, or forensic authenticity.



\## Citation



If you use EditTrace-XAI in academic work, please cite the accompanying paper.



```bibtex

@inproceedings{edittrace\_xai,

&#x20; title     = {EditTrace-XAI: Action-Level Debugging of Image Editing Workflows},

&#x20; author    = {TBD},

&#x20; booktitle = {TBD},

&#x20; year      = {TBD}

}

```



The citation will be updated with the final publication metadata.



\## License



See the `LICENSE` file for the terms under which the code in this repository is distributed.



