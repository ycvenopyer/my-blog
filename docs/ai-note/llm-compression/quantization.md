# Quantization

Quantization is the process of converting the parameters and activations of a model to a lower precision. For example, weights stored using 16 bits can be converted into a 4-bit representation. This technique has become increasingly important to reduce the computational and memory costs associated with LLMs.

Base techniques: Learn the different levels of precision (FP32, FP16, INT8, etc.) and how to perform naïve quantization with absmax and zero-point techniques.

GGUF & llama.cpp: Originally designed to run on CPUs, llama.cpp and the GGUF format have become the most popular tools to run LLMs on consumer-grade hardware. It supports storing special tokens, vocabulary, and metadata in a single file.

GPTQ & AWQ: Techniques like GPTQ/EXL2 and AWQ introduce layer-by-layer calibration that retains performance at extremely low bitwidths. They reduce catastrophic outliers using dynamic scaling, selectively skipping or re-centering the heaviest parameters.

SmoothQuant & ZeroQuant: New quantization-friendly transformations (SmoothQuant) and compiler-based optimizations (ZeroQuant) help mitigate outliers before quantization. They also reduce hardware overhead by fusing certain ops and optimizing dataflow.
