# Edge-QLoRA (Work in Progress)

**FPGA-Accelerated Fine-Tuning of Quantized LLMs for Edge Devices**

Deploying and fine-tuning large language models (LLMs) on edge devices is a major challenge due to their high computational demands, memory usage, and power consumption. While **Low-Rank Adaptation (LoRA)** helps reduce fine-tuning complexity, its memory footprint is still too large for edge environments. **Quantized LoRA (QLoRA)** addresses this by combining 4-bit quantization with LoRA, enabling significant memory savings with minimal loss in model accuracy.

This project—**Edge-QLoRA**—aims to implement an FPGA-accelerated version of QLoRA, targeting low-power, real-time fine-tuning of LLMs directly on edge devices. The project is currently under development as part of a competitive accelerator program.

### 🎯 Project Goals

* ✅ Develop a memory-efficient, hardware-accelerated implementation of QLoRA on FPGA
* ⚙️ Design custom IP cores for 4-bit quantization and dequantization
* 🚀 Optimize matrix operations using systolic arrays and tiling for parallelism and efficient on-chip memory usage
* 🔋 Create energy-efficient pipelines suitable for real-time inference and fine-tuning

### 📊 Expected Outcomes

* Reduce the memory footprint of TinyLlama (1.1B params) to under 300MB
* Achieve significant speedups in quant/dequant operations over software baselines
* Enable fine-tuning of LLMs on edge hardware with up to 75% reduction in model size (vs. FP16)
* Validate performance through simulations 

### 🛠 Status

This repository contains early documentation, architectural planning, and initial Python-based prototypes. Hardware implementation and benchmarking are ongoing.

### 🌍 Target Applications

* Privacy-preserving healthcare inference
* On-device AI for mobile and wearable applications
* Smart IoT edge processing

### 📅 What's Next

* Hardware prototyping on FPGA
* Dynamic quantization strategy exploration
* Support for larger LLMs (e.g., Mistral, Phi)

