# Quantum Vision Transformer (QViT) Architecture Sketch

## Classical Vision Transformer Recap
- Patch embedding: splits image into patches, projects to embedding space
- Transformer encoder: self-attention layers process patch embeddings
- Classification head: uses CLS token for prediction

## Quantum Extension Ideas
- **Patch Encoding:** Encode each image patch into a quantum state using amplitude encoding or basis encoding.
- **Quantum Attention:** Replace classical self-attention with quantum circuits that entangle patch qubits and perform quantum operations to simulate attention.
- **Quantum Transformer Encoder:** Use parameterized quantum circuits (PQCs) for each layer, with entanglement between patch qubits to capture global information.
- **Quantum Readout:** Measure qubits to obtain patch-level or global features, then use classical or quantum post-processing for classification.

## Example QViT Architecture
1. **Input:** MNIST image split into patches
2. **Patch Quantum Encoding:** Each patch encoded into a set of qubits
3. **Quantum Attention Layer:** Quantum circuit entangles patch qubits, parameterized gates simulate attention
4. **Quantum Transformer Layer:** Multiple layers of PQCs, each with entanglement and local rotations
5. **Readout:** Measure qubits, aggregate results for classification

## Diagram
```
Image → Patch Split → Quantum Encoding → Quantum Attention → Quantum Transformer Layers → Measurement → Classification
```

## Comments
- Quantum circuits can exploit entanglement and superposition for richer feature extraction
- Quantum attention may be implemented via controlled gates and parameterized rotations
- Hybrid classical-quantum models can combine quantum feature extraction with classical classification
- Scalability depends on available qubits and quantum hardware

---
See https://arxiv.org/abs/2205.06217 and https://arxiv.org/abs/2210.08566 for more details.
