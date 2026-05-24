

# AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs

This repository contains the official implementation of **AGoQ**, a memory-efficient distributed training system for Large Language Models (LLMs). AGoQ significantly reduces GPU memory requirements by integrating layer-aware activation quantization and precision-preserved gradient quantization.

## Overview

Training large language models typically demands massive GPU memory, with activations and gradients occupying a substantial portion. While existing quantization approaches often struggle with 4-bit activations and 8-bit gradients due to slow convergence or accuracy loss, AGoQ introduces a practical solution to push quantization further.

### Key Features

* 
**Layer-Aware Activation Quantization (LAAQ):** Dynamically allocates appropriate bit-widths for the activations of various layers based on their specific types and pipeline stages. This approach achieves near 4-bit activation storage without sacrificing accuracy.


* 
**Dynamic Bit-width Compensation (DBCA-PP):** Leverages imbalanced memory usage across Interleaved 1F1B pipeline parallelism stages. Devices storing fewer activation batches are assigned higher quantization bit-widths, making full use of under-utilized memory to compensate for precision loss.


* 
**Precision-Preserved Gradient Quantization (QuanGrad):** Employs 8-bit (FP8) format to store gradients for local accumulation and utilizes a precision-preserving 8-bit All-Reduce communication strategy to minimize memory usage and communication overhead.


* 
**Kernel Fusion:** Fuses activation quantization and dequantization with nearby GEMM operations into a single GPU kernel, effectively eliminating the computation overheads typically introduced by quantization.


* 
**Hardware Support:** Compatible with NVIDIA GPUs and Huawei Ascend NPUs.



## Performance Highlights

Extensive experiments on LLMs ranging from 8B to 34B parameters (including LLaMA2, LLaMA3, and CodeLLaMA) demonstrate that AGoQ delivers significant improvements:

* Reduces memory footprint by up to **52%** compared to standard full-precision training.


* Achieves up to a **1.34x** improvement in end-to-end training speed over state-of-the-art systems like Megatron-LM (w/ or w/o ZeRO), DeepSpeed, and COAT.


* Maintains strict convergence loss during pretraining and zero-shot accuracy on downstream tasks.



## Citation

If you find this work helpful in your research, please consider citing our paper:

```bibtex
@article{lin2026agoq,
  title={AGoQ: Activation and Gradient Quantization for Memory-Efficient Distributed Training of LLMs},
  author={Lin, Wenxiang and Huang, Juntao and Zhang, Luhan and Li, Laili and Bao, Xiang and Zhang, Mengyang and Wang, Bing and Shi, Shaohuai},
  journal={arXiv preprint arXiv:2605.00539},
  year={2026}
}

```

## Acknowledgments

This project is built upon [MindSpeed-LLM](https://gitee.com/ascend/MindSpeed-LLM/). We sincerely thank the developers for their foundational work.
