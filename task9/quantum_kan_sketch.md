# Quantum Kolmogorov-Arnold Network (QKAN) Architecture Sketch

## Classical KAN Recap
- Basis-spline layers for nonlinear function approximation
- Fully connected layers for feature extraction

## Quantum Extension Ideas
- **Quantum Spline Encoding:** Encode input features into quantum states, use quantum gates to simulate spline basis functions
- **Quantum Nonlinearity:** Parameterized quantum circuits (PQCs) for nonlinear transformations, inspired by spline layers
- **Quantum Feature Extraction:** Entanglement and superposition for richer feature extraction
- **Quantum Readout:** Measure qubits to obtain features, use classical or quantum post-processing for classification

## Example QKAN Architecture
1. **Input:** MNIST image encoded as quantum state
2. **Quantum Spline Layer:** PQC simulates basis-spline transformation
3. **Quantum Feature Layer:** Entanglement and local rotations
4. **Readout:** Measurement for classification

## Diagram
```
Image → Quantum Encoding → Quantum Spline PQC → Quantum Feature PQC → Measurement → Classification
```

## Comments
- Quantum circuits can approximate nonlinear functions via parameterized gates
- Quantum splines may be implemented using controlled rotations and entanglement
- Hybrid classical-quantum models can combine quantum feature extraction with classical classification
- Scalability depends on available qubits and quantum hardware

---
See https://arxiv.org/abs/2205.06217 and https://arxiv.org/abs/2210.08566 for more details.
