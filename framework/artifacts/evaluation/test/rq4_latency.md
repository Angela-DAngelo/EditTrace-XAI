# RQ4 latency summary

Times exclude model construction and warm-up. GPU evaluators synchronize before and after timed inference.

| Component | Model | History | n | Median (ms) | p95 (ms) | Calls median | Calls p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| diagnosis | torchvision:resnet50:IMAGENET1K_V2 | all | 601 | 170.715 | 272.632 | 5.0 | 7.0 |
| diagnosis | torchvision:vit_b_16:IMAGENET1K_V1 | all | 619 | 303.228 | 469.291 | 5.0 | 7.0 |
| interaction | torchvision:resnet50:IMAGENET1K_V2 | all | 640 | 537.439 | 814.134 | 16.0 | 22.0 |
| interaction | torchvision:vit_b_16:IMAGENET1K_V1 | all | 571 | 1002.369 | 1539.347 | 16.0 | 22.0 |
| repair | torchvision:resnet50:IMAGENET1K_V2 | all | 304 | 237.498 | 5329.257 | 5.0 | 126.0 |
| repair | torchvision:vit_b_16:IMAGENET1K_V1 | all | 266 | 394.075 | 9538.804 | 5.0 | 126.0 |

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
