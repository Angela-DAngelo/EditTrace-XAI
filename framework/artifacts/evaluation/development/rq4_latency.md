# RQ4 latency summary

Times exclude model construction and warm-up. GPU evaluators synchronize before and after timed inference.

| Component | Model | History | n | Median (ms) | p95 (ms) | Calls median | Calls p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| diagnosis | torchvision:resnet50:IMAGENET1K_V2 | all | 158 | 172.514 | 273.252 | 5.0 | 7.0 |
| diagnosis | torchvision:vit_b_16:IMAGENET1K_V1 | all | 156 | 293.353 | 477.808 | 5.0 | 7.0 |
| interaction | torchvision:resnet50:IMAGENET1K_V2 | all | 161 | 515.610 | 853.907 | 16.0 | 22.0 |
| interaction | torchvision:vit_b_16:IMAGENET1K_V1 | all | 144 | 933.055 | 1547.897 | 16.0 | 22.0 |
| repair | torchvision:resnet50:IMAGENET1K_V2 | all | 81 | 250.208 | 4966.305 | 6.0 | 126.0 |
| repair | torchvision:vit_b_16:IMAGENET1K_V1 | all | 74 | 376.229 | 7487.024 | 5.0 | 108.5 |

## Hardware

```json
{
  "cuda_available": false,
  "cuda_version": "13.0",
  "cudnn_version": null,
  "gpu": null,
  "platform": "Linux-5.15.0-173-generic-x86_64-with-glibc2.35",
  "processor": "x86_64",
  "python": "3.11.15",
  "torch": "2.13.0+cu130"
}
```
