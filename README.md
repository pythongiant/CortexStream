
# 🧠 CortexStream

**High-performance LLM inference runtime for Apple Silicon — continuous batching, KV-cache orchestration, and streaming generation powered by MLX + Metal.**

CortexStream brings **Triton-LLM style intelligence** to the Apple ecosystem.
It focuses on the *real bottlenecks* of large language model serving:

* Efficient **continuous batching**
* Robust **KV cache allocation**
* Deterministic **scheduling**
* **Streaming token** delivery
* Minimal overhead compute pipeline on **MLX + MPS**

Designed to demonstrate that Apple Silicon can serve LLMs *not just locally… but seriously*.